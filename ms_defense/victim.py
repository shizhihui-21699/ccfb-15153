"""Victim risk-control model: gradient-boosted decision tree served via API."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from . import config
from .data import Dataset


def train_victim(ds: Dataset) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=400,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=config.RANDOM_STATE,
    )
    model.fit(ds.X_train, ds.y_train)
    return model


def evaluate(model, ds: Dataset) -> dict[str, float]:
    p = model.predict_proba(ds.X_test)[:, 1]
    pred = (p >= config.RISK_THRESHOLD).astype(int)
    return {
        "auc": roc_auc_score(ds.y_test, p),
        "auprc": average_precision_score(ds.y_test, p),
        "accuracy": accuracy_score(ds.y_test, pred),
        "base_rate": float(ds.y_test.mean()),
    }


def save_model(model, ds: Dataset, path: Path | None = None) -> Path:
    path = path or config.MODEL_DIR / f"victim_{ds.name}.pkl"
    with open(path, "wb") as fh:
        pickle.dump({"model": model, "feature_names": ds.feature_names,
                     "scaler": ds.scaler}, fh)
    return path


def load_model(path: Path | None = None) -> tuple[HistGradientBoostingClassifier, list[str], StandardScaler]:
    if path is None:
        candidates = sorted(config.MODEL_DIR.glob("victim_*.pkl"))
        if not candidates:
            raise FileNotFoundError("no trained victim model found; run scripts/train_victim.py")
        path = candidates[0]
    with open(path, "rb") as fh:
        bundle = pickle.load(fh)
    return bundle["model"], bundle["feature_names"], bundle["scaler"]
