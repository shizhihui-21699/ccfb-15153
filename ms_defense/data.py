"""Data layer.

Primary dataset: UCI Statlog German Credit (public, 1000 rows, 24 numeric
features, binary credit-risk label; class 2 in the raw file = bad credit and
is remapped to label 1 = risky).

Fallback: a synthetic credit-risk dataset with similar class imbalance, used
only when the network is unavailable so the lab remains reproducible offline.
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config


@dataclass
class Dataset:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler
    name: str

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def _download_german() -> pd.DataFrame:
    cache = config.DATA_DIR / "german.data-numeric"
    if not cache.exists():
        req = urllib.request.Request(config.GERMAN_URL, headers={"User-Agent": "ms-defense-lab/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            cache.write_bytes(resp.read())
    rows = []
    for line in cache.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 25:
            rows.append([float(x) for x in parts[:25]])
    df = pd.DataFrame(rows)
    # last column is the label: 1 = good, 2 = bad -> 0/1 with 1 = risky
    X = df.iloc[:, :-1].to_numpy()
    y = (df.iloc[:, -1].to_numpy() == 2).astype(int)
    names = [f"f{i}" for i in range(X.shape[1])]
    return X, y, names


def _synthetic(n_samples: int = 4_000, seed: int = 0):
    """Synthetic imbalanced credit-risk data, ~25% positives like German."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=24,
        n_informative=10,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        weights=[0.7, 0.3],
        n_clusters_per_class=3,
        class_sep=1.1,
        flip_y=0.08,
        random_state=seed,
    )
    names = [f"f{i}" for i in range(X.shape[1])]
    return X, y, names


def load_dataset(name: str | None = None) -> Dataset:
    name = name or config.DATASET
    try:
        X, y, names = _download_german()
        source = "german"
    except Exception as exc:  # offline / mirror blocked
        print(f"[data] German Credit unavailable ({exc}); using synthetic data")
        X, y, names = _synthetic(seed=config.RANDOM_STATE)
        source = "synthetic"

    idx = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=config.TEST_SIZE, stratify=y[idx],
        random_state=config.RANDOM_STATE,
    )
    val_frac = config.VAL_SIZE / (1 - config.TEST_SIZE)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac, stratify=y[idx_trainval],
        random_state=config.RANDOM_STATE,
    )
    scaler = StandardScaler().fit(X[idx_train])
    return Dataset(
        X_train=scaler.transform(X[idx_train]),
        X_val=scaler.transform(X[idx_val]),
        X_test=scaler.transform(X[idx_test]),
        y_train=y[idx_train],
        y_val=y[idx_val],
        y_test=y[idx_test],
        feature_names=names,
        scaler=scaler,
        name=source,
    )


def inverse_scale(ds: Dataset, X_scaled: np.ndarray) -> np.ndarray:
    """Return points in the original (unscaled) feature space."""
    return ds.scaler.inverse_transform(X_scaled)
