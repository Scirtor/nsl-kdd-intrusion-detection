"""Select on a fixed training-file validation split, then evaluate official test."""
from pathlib import Path
import argparse
import json
import platform
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier
import xgboost

from nslkdd.data import ROOT, load_dataset, download, checksum
from nslkdd.features import build_pipeline
from nslkdd.report import plot_report

SEED = 42


def metrics(y, predictions, scores):
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        'accuracy': float(accuracy_score(y, predictions)),
        'precision': float(precision_score(y, predictions, zero_division=0)),
        'recall': float(recall_score(y, predictions, zero_division=0)),
        'f1': float(f1_score(y, predictions, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        'false_positive_rate': float(fp / (fp + tn)) if fp + tn else None,
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def evaluate(model, X, y):
    predictions = model.predict(X)
    scores = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X)
    return metrics(y, predictions, scores), predictions


def run(data_dir, output_dir, jobs=2):
    output_dir.mkdir(parents=True, exist_ok=False)
    download(data_dir)
    X, y, attacks = load_dataset(data_dir / 'KDDTrain+.txt')
    fit_ids, valid_ids = train_test_split(np.arange(len(X)), test_size=.2, stratify=y, random_state=SEED)
    pd.DataFrame({'row': np.arange(len(X)), 'split': np.where(np.isin(np.arange(len(X)), valid_ids), 'validation', 'fit')}).to_csv(output_dir / 'split.csv', index=False)
    estimators = {
        'Majority baseline': DummyClassifier(strategy='most_frequent'),
        'Logistic Regression': LogisticRegression(max_iter=2000, C=1, random_state=SEED),
        'Random Forest': RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=jobs),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=jobs),
        'XGBoost': XGBClassifier(n_estimators=300, max_depth=6, learning_rate=.1, subsample=.9,
                                colsample_bytree=.9, eval_metric='logloss', random_state=SEED,
                                n_jobs=jobs, tree_method='hist'),
        'Linear SVM': LinearSVC(C=1, max_iter=10000, dual=False, random_state=SEED),
    }
    rows = []
    for name, estimator in estimators.items():
        print(f'Fitting {name}', flush=True)
        model = build_pipeline(estimator)
        start = time.perf_counter()
        model.fit(X.iloc[fit_ids], y.iloc[fit_ids])
        result, _ = evaluate(model, X.iloc[valid_ids], y.iloc[valid_ids])
        rows.append({'model': name, **result, 'fit_and_evaluate_seconds': round(time.perf_counter()-start, 3)})
        print(f'  validation F1: {result["f1"]:.5f}', flush=True)
    # Fixed order breaks an exact tie. No official-test metric is used here.
    selected = max(rows, key=lambda r: r['f1'])['model']
    model = build_pipeline(estimators[selected])
    print(f'Refitting {selected} on all KDDTrain+ records', flush=True)
    model.fit(X, y)
    # Load the official test only after model selection and full-train fitting.
    Xt, yt, test_attacks = load_dataset(data_dir / 'KDDTest+.txt')
    test_metrics, predictions = evaluate(model, Xt, yt)
    train_hashes = set(pd.util.hash_pandas_object(X, index=False))
    overlap = pd.util.hash_pandas_object(Xt, index=False).isin(train_hashes).to_numpy()
    no_overlap_metrics = None
    if (~overlap).any():
        no_overlap_metrics, _ = evaluate(model, Xt.loc[~overlap], yt.loc[~overlap])
    per_attack = []
    for attack in sorted(test_attacks.unique()):
        mask = (test_attacks == attack).to_numpy()
        per_attack.append({'attack_type': attack, 'records': int(mask.sum()),
                           'correct_binary_classification_rate': float((predictions[mask] == yt.to_numpy()[mask]).mean()),
                           'seen_in_training': bool(attack in set(attacks))})
    pd.DataFrame(per_attack).to_csv(output_dir / 'per-attack.csv', index=False)
    comparison = pd.DataFrame([{k:v for k,v in r.items() if k != 'confusion_matrix'} for r in rows])
    comparison.to_csv(output_dir / 'validation.csv', index=False)
    report = {
        'seed': SEED, 'selection': 'maximum attack-class F1 on a fixed stratified 20% holdout of KDDTrain+',
        'threshold': 'unmodified estimator default; no test-set threshold tuning',
        'selected_model': selected, 'fit_rows': len(fit_ids), 'validation_rows': len(valid_ids),
        'final_training_rows': len(X), 'official_test_rows': len(Xt),
        'original_features': X.shape[1], 'engineered_features': 2,
        'validation_results': rows, 'official_test': test_metrics,
        'exact_feature_overlap_test_rows': int(overlap.sum()),
        'official_test_without_exact_training_feature_matches': no_overlap_metrics,
        'data_sha256': {name: checksum(data_dir/name) for name in ['KDDTrain+.txt','KDDTest+.txt']},
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                     'scikit-learn': sklearn.__version__, 'xgboost': xgboost.__version__, 'joblib': joblib.__version__},
    }
    (output_dir / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    joblib.dump(model, output_dir / 'model.joblib')
    predictions_frame = pd.DataFrame({'test_row': np.arange(len(yt)), 'actual': yt, 'prediction': predictions,
                                      'attack_type': test_attacks, 'matches_training_features': overlap})
    predictions_frame.to_csv(output_dir / 'test-predictions.csv', index=False)
    plot_report(report, output_dir)
    print(json.dumps({'selected_model':selected,'official_test':test_metrics},indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--output-dir',type=Path,default=ROOT/'runs/baseline')
    p.add_argument('--jobs',type=int,default=2)
    args=p.parse_args()
    if args.jobs < 1:p.error('--jobs must be positive')
    with threadpool_limits(limits=args.jobs):
        run(args.data_dir,args.output_dir,args.jobs)
