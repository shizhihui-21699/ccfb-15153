"""Surrogate training and the metrics that quantify extraction success."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score


def train_surrogate(X: np.ndarray, y_score: np.ndarray,
                    kind: str = "gbt", seed: int = 0):
    """Train the attacker's substitute model on stolen (x, score) pairs.

    GBT matches the victim family; MLP represents an attacker using a
    different model class — tests whether extraction transfers across
    architectures.
    """
    # distillation target: thresholded hard labels when scores are quantized
    y_hard = (y_score >= 0.5).astype(int)
    if kind == "gbt":
        model = HistGradientBoostingClassifier(
            max_iter=300, max_leaf_nodes=15, learning_rate=0.08,
            min_samples_leaf=15, random_state=seed,
        )
        model.fit(X, y_hard)
    elif kind == "mlp":
        model = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                          max_iter=400, early_stopping=True,
                          random_state=seed),
        )
        model.fit(X, y_hard)
    else:
        raise ValueError(kind)
    return model


def fidelity(surrogate, victim, X_test: np.ndarray, y_test: np.ndarray,
             threshold: float = 0.5) -> dict[str, float]:
    """How close is the stolen model to the victim?"""
    p_v = victim.predict_proba(X_test)[:, 1]
    p_s = surrogate.predict_proba(X_test)[:, 1]
    lab_v = (p_v >= threshold).astype(int)
    lab_s = (p_s >= threshold).astype(int)
    return {
        "score_rmse": float(np.sqrt(np.mean((p_v - p_s) ** 2))),
        "score_mae": float(np.mean(np.abs(p_v - p_s))),
        "label_agreement": float(np.mean(lab_v == lab_s)),
        "label_flip_vs_victim": float(np.mean(lab_v != lab_s)),
        "spearman": float(np.corrcoef(np.argsort(np.argsort(p_v)),
                                      np.argsort(np.argsort(p_s)))[0, 1]),
        "surrogate_auc": roc_auc_score(y_test, p_s),
        "victim_auc": roc_auc_score(y_test, p_v),
        "surrogate_auprc": average_precision_score(y_test, p_s),
        "victim_auprc": average_precision_score(y_test, p_v),
        "auc_gap": roc_auc_score(y_test, p_v) - roc_auc_score(y_test, p_s),
    }


def evasion_bypass(surrogate, victim, X_risky: np.ndarray, l2_budget: float = 2.0,
                   n_restarts: int = 6, seed: int = 0) -> dict[str, float]:
    """White-box-style evasion on the *surrogate*, evaluated against the
    *victim* — the attacker's end goal. Coordinate-descent perturbations
    bounded by an L2 budget; reports transfer rate to the victim.
    """
    rng = np.random.default_rng(seed)
    n, d = X_risky.shape
    X_adv = X_risky.copy()
    l2_used = np.zeros(n)
    for i in range(n):
        x = X_risky[i].copy()
        best_x, best_p = x.copy(), float(surrogate_predict(surrogate, x)[0])
        for r in range(n_restarts):
            cand = best_x.copy()
            n_pert = int(rng.integers(2, max(3, d // 2)))
            steps = rng.choice(d, size=min(n_pert, d), replace=False)
            for c in steps:
                cand[c] += rng.choice([-1, 1]) * rng.uniform(0.05, 0.4)
            if np.linalg.norm(cand - X_risky[i]) > l2_budget:
                continue
            p = float(surrogate_predict(surrogate, cand)[0])
            if p < best_p:
                best_p, best_x = p, cand
        X_adv[i] = best_x
        l2_used[i] = np.linalg.norm(best_x - X_risky[i])
    p_victim = victim.predict_proba(X_adv)[:, 1]
    p_clean = victim.predict_proba(X_risky)[:, 1]
    return {
        "bypass_rate_vs_victim": float(np.mean(p_victim < 0.5)),
        "score_drop_on_victim": float(np.mean(p_clean - p_victim)),
        "mean_l2": float(l2_used.mean()),
        "fraction_within_budget": float(np.mean(l2_used <= l2_budget)),
    }


def surrogate_predict(surrogate, x: np.ndarray) -> np.ndarray:
    return surrogate.predict_proba(np.atleast_2d(x))[:, 1]
