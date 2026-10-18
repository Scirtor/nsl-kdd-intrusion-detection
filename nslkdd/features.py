"""Stateless engineering followed by fold-local learned preprocessing."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nslkdd.data import CATEGORICAL, NUMERIC, validate_features


class FeatureEngineer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        validate_features(X)
        return self

    def transform(self, X):
        result = validate_features(X)
        result['bytes_ratio'] = result.src_bytes / (result.dst_bytes + 1)
        result['error_rate_total'] = result.serror_rate + result.rerror_rate
        if np.isinf(result[NUMERIC + ['bytes_ratio', 'error_rate_total']]).any().any():
            raise ValueError('Engineered features overflowed; inspect numeric input ranges')
        return result


def build_pipeline(estimator):
    numeric = Pipeline([
        ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scale', StandardScaler()),
    ])
    preprocessing = ColumnTransformer([
        ('numeric', numeric, NUMERIC + ['bytes_ratio', 'error_rate_total']),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
    ], remainder='drop')
    return Pipeline([('engineer', FeatureEngineer()), ('preprocess', preprocessing), ('model', estimator)])
