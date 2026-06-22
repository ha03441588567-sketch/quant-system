"""
Ensemble prediction layer: XGBoost + LightGBM -> logistic-regression meta-learner (stacking).
Uses walk-forward splits (NOT random k-fold) because shuffling time-series leaks future
information into training — this is the #1 reason naive crypto ML backtests look amazing
and then fail live.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import config
from data_pipeline import FEATURE_COLUMNS


def walk_forward_splits(n_rows: int, n_splits: int = None):
    """Yields (train_idx, test_idx) where test always comes strictly after train in time."""
    n_splits = n_splits or config.N_WALKFORWARD_SPLITS
    fold_size = n_rows // (n_splits + 1)

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_end = fold_size * (i + 1)
        yield np.arange(0, train_end), np.arange(train_end, min(test_end, n_rows))


class EnsembleModel:
    def __init__(self):
        self.xgb = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        )
        self.lgbm = LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, verbose=-1,
        )
        self.meta = LogisticRegression()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.xgb.fit(X, y)
        self.lgbm.fit(X, y)

        # Meta-learner trains on the base models' own (in-sample) predictions.
        # For a production system, generate these via nested walk-forward instead of in-sample
        # predictions, to avoid meta-learner overfitting — flagged here, not yet implemented.
        stacked = np.column_stack([
            self.xgb.predict_proba(X)[:, 1],
            self.lgbm.predict_proba(X)[:, 1],
        ])
        self.meta.fit(stacked, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        stacked = np.column_stack([
            self.xgb.predict_proba(X)[:, 1],
            self.lgbm.predict_proba(X)[:, 1],
        ])
        return self.meta.predict_proba(stacked)[:, 1]

    def save(self, path: str):
        if not self.is_fitted:
            raise RuntimeError("Refusing to save an unfitted model")
        joblib.dump({"xgb": self.xgb, "lgbm": self.lgbm, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: str) -> "EnsembleModel":
        bundle = joblib.load(path)
        model = cls()
        model.xgb = bundle["xgb"]
        model.lgbm = bundle["lgbm"]
        model.meta = bundle["meta"]
        model.is_fitted = True
        return model


def run_walk_forward_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """Trains+tests across multiple time-ordered folds. Returns per-fold metrics —
    look at the SPREAD across folds, not just the average. Wide spread = unstable model."""
    X = df[FEATURE_COLUMNS]
    y = df["target"]

    results = []
    for fold_n, (train_idx, test_idx) in enumerate(walk_forward_splits(len(df)), start=1):
        if len(test_idx) == 0:
            continue
        model = EnsembleModel()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])

        proba = model.predict_proba(X.iloc[test_idx])
        preds = (proba > 0.5).astype(int)

        acc = accuracy_score(y.iloc[test_idx], preds)
        try:
            auc = roc_auc_score(y.iloc[test_idx], proba)
        except ValueError:
            auc = float("nan")

        results.append({"fold": fold_n, "accuracy": acc, "auc": auc, "n_test": len(test_idx)})

    return pd.DataFrame(results)
