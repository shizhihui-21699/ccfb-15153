"""Detection evaluation: label windows by account ground truth and compute
precision / recall / FPR / ROC-AUC for the rule-based and ML detectors."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ms_defense import config
from ms_defense.detection import RuleBasedDetector, MLDetector
from ms_defense.detection_features import WindowFeatureExtractor


def collect_window_samples(extractor: WindowFeatureExtractor,
                           benign_frames: list[pd.DataFrame],
                           attack_frames: dict[str, pd.DataFrame],
                           distributed_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Replay audit frames through fresh extractors and collect one labeled
    feature vector per sliding window."""
    X_list, y_list, meta = [], [], []

    def replay(frame: pd.DataFrame, label: int, tag: str):
        feat_cols = [c for c in frame.columns if c.startswith("f")]
        for acct, g in frame.groupby("account_id"):
            g = g.sort_values("ts")
            X = g[feat_cols].to_numpy()
            scores = g["score"].to_numpy()
            labels = g["label"].to_numpy()
            ts = g["ts"].to_numpy()
            for start in range(0, len(X) - config.WINDOW_SIZE + 1, config.WINDOW_STRIDE):
                sl = slice(start, start + config.WINDOW_SIZE)
                feats = extractor.extract(X[sl], scores[sl], labels[sl], ts[sl])
                X_list.append(feats)
                y_list.append(label)
                meta.append({"account": acct, "tag": tag, "label": label})

    for i, fr in enumerate(benign_frames):
        replay(fr, 0, f"benign_{i}")
    for name, fr in attack_frames.items():
        replay(fr, 1, name)
    replay(distributed_frame, 1, "distributed")
    return np.vstack(X_list), np.array(y_list), meta


def evaluate_detectors(X: np.ndarray, y: np.ndarray,
                       extractor: WindowFeatureExtractor) -> tuple[dict, MLDetector]:
    from sklearn.metrics import (
        precision_recall_fscore_support, roc_auc_score, confusion_matrix,
    )
    results = {}

    # rule-based score = weighted feature scorer applied directly
    rule = RuleBasedDetector(extractor)
    rule_scores = np.array([rule._score_window(row) for row in X])
    rule_pred = (rule_scores >= config.DETECT_ALERT).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, rule_pred, average="binary",
                                                   zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, rule_pred, labels=[0, 1]).ravel()
    results["rule_based"] = {
        "precision": p, "recall": r, "f1": f1,
        "fpr": fp / (tn + fp + 1e-12),
        "roc_auc": roc_auc_score(y, rule_scores),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }

    # supervised ML detector, cross-fit style split on windows
    ml = MLDetector(extractor)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    cut = int(0.6 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    ml.fit(X[tr], y[tr])
    ml_scores = np.array([ml.clf.predict_proba(row.reshape(1, -1))[0, 1] for row in X[te]])
    ml_pred = (ml_scores >= config.DETECT_ALERT).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y[te], ml_pred, average="binary",
                                                   zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y[te], ml_pred, labels=[0, 1]).ravel()
    results["ml_random_forest"] = {
        "precision": p, "recall": r, "f1": f1,
        "fpr": fp / (tn + fp + 1e-12),
        "roc_auc": roc_auc_score(y[te], ml_scores),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
    # feature importance for interpretability
    results["ml_feature_importance"] = dict(
        sorted(zip(extractor.FEATURE_NAMES, ml.clf.feature_importances_),
               key=lambda kv: -kv[1])
    )
    return results, ml
