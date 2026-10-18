# Evaluation protocol and limitations

## How a model is chosen

The published run fixes seed 42 and uses all 125,973 KDDTrain+ rows. A stratified split reserves 25,195 rows for validation and 100,778 for fitting. Split membership is exported to the run directory. All six candidates use the same rows. The majority predictor is included as a baseline, alongside Logistic Regression, Random Forest, KNN, XGBoost and Linear SVM.

The input contains 41 original features. `attack_type`, the derived binary target and `difficulty` are excluded. Stateless feature engineering adds `src_bytes / (dst_bytes + 1)` and `serror_rate + rerror_rate`. Numeric median imputation and standardisation, plus one-hot categorical encoding, are learned inside each fitted pipeline. Unknown categories are ignored without refitting the encoder.

The candidate with the highest attack-class validation F1 is selected; fixed candidate order breaks an exact tie. The selected estimator is then fitted on the full training file. KDDTest+ is loaded for evaluation only after that step. Estimator defaults determine the prediction threshold; it is not tuned on the test file. This is a single fixed validation split, not cross-validation or an exhaustive hyperparameter search.

## Published run

XGBoost was selected with validation F1 0.9993. On 22,544 official test records, its attack recall is 66.01%, precision 96.89% and F1 0.7852. There are 4,362 missed attacks and 272 false alarms. The false positive rate is 2.80% of normal records.

These results show why a near-perfect random holdout score should not be used as the headline detection quality. `results/per-attack.csv` breaks down correct binary classification by named attack and marks whether that attack name occurs in the training file. This is a binary classifier: it does not predict the attack family.

The file audit found 664 test rows with matching 41-feature vectors in KDDTrain+. The primary score keeps the published split intact. The additional score without those rows is F1 0.7760 and attack recall 64.71%. It is a diagnostic, not a second model selection round.

## What changed from the original notebook

The [original academic notebook](../legacy/final_nslkdd.ipynb) is retained as historical evidence, with cell outputs removed. It uses a random holdout of KDDTrain+ and does not evaluate the official KDDTest+ file. Its categorical encoding and median imputation are fitted before the train/test split; its cross-validation reuses already scaled data. The final RF/XGBoost deployment choice is based on that holdout's F1.

The new pipeline moves learned preprocessing inside fitting, uses validation for the deployment choice and evaluates a separate official test file after selection. Original results and the new official-test results are different experiments and should not be compared as if they used the same evaluation set. The original saved pickle files are neither loaded nor redistributed here.

## Limits

- NSL-KDD is a historical benchmark. [Its original distributor](https://www.unb.ca/cic/datasets/nsl.html) explicitly notes limitations in representing real networks. These numbers do not establish performance on modern production traffic.
- A single split and fixed seed do not measure uncertainty across repetitions. No confidence interval, temporal generalisation claim or zero-day detection claim is made.
- Binary classification can hide large differences between attack types. Inspect per-attack counts, missed attacks and the confusion matrix alongside aggregate scores.
- The demo expects the benchmark's engineered connection schema. It does not create those fields from PCAP files or validate an operational collection pipeline.
- Matching feature vectors are audited, but removing them does not remove all statistical similarities between the files.
- The published run records software versions and data hashes. Different dependency builds can still produce small numerical differences. No hardware-independent bitwise reproducibility is claimed.

## Outputs and traceability

`metrics.json` records selection criteria, row counts, feature count, hashes, package versions and both official-test evaluations. `validation.csv` contains candidate scores and timings. `per-attack.csv` contains the attack-level diagnostic. A local run additionally writes the model, split assignments and test predictions; these generated files are ignored by Git.
