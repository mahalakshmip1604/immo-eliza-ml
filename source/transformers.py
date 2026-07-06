# transformers.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """
    Fills missing values in `target_cols` using the median of each row's
    `group_col` group. Medians are learned only from the data passed to
    `fit`, so when used inside a Pipeline it only ever sees the training
    fold/split - no leakage from validation/test data.
    """

    def __init__(self, group_col: str, target_cols: list[str]):
        self.group_col = group_col
        self.target_cols = target_cols

    def fit(self, X: pd.DataFrame, y=None):
        self.group_medians_ = {
            col: X.groupby(self.group_col)[col].median() for col in self.target_cols
        }
        self.global_medians_ = X[self.target_cols].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.target_cols:
            mapped = X[self.group_col].map(self.group_medians_[col])
            X[col] = X[col].fillna(mapped)
            X[col] = X[col].fillna(self.global_medians_[col])
        return X