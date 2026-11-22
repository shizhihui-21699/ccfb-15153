"""Generate figures from the experiment artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ms_defense import config

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})


def _load():
    fid = pd.read_csv(config.REPORT_DIR / "fidelity_curves.csv")
    eva = pd.read_csv(config.REPORT_DIR / "evasion_summary.csv")
    summary = json.loads((config.REPORT_DIR / "run_summary.json").read_text())
    return fid, eva, summary


def fig_fidelity_vs_budget(fid: pd.DataFrame):
    attacks = ["uniform_random", "gaussian_mimicry", "boundary_search",
               "label_only_boundary"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    metrics = [("label_agreement", "Label agreement with victim"),
               ("surrogate_auc", "Surrogate ROC-AUC"),
               ("score_rmse", "Score RMSE (lower=better)"),
               ("evasion_bypass_rate_vs_victim", "Evasion bypass rate vs victim")]
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for atk in attacks:
            for mode, style in [("undefended", "o-"), ("defended", "s--")]:
                sub = fid[(fid["attack"] == atk) & (fid["mode"] == mode)] \
                    .sort_values("budget")
                if sub.empty:
                    continue
                ax.plot(sub["budget"], sub[metric], style,
                        label=f"{atk} [{mode}]", markersize=4, alpha=0.85)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xlabel("stolen queries")
        ax.legend(fontsize=6, ncol=2)
    fig.suptitle("Extraction fidelity: undefended vs. adaptively defended API",
                 fontsize=12)
    fig.tight_layout()
    out = config.FIG_DIR / "fidelity_vs_budget.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_detection(summary: dict):
    det = summary["detection"]
    names = [k for k in det if k != "ml_feature_importance"]
    metrics = ["precision", "recall", "f1", "fpr", "roc_auc"]
    vals = np.array([[det[n][m] for m in metrics] for n in names])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(metrics))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        ax.bar(x + i * w, vals[i], w, label=n)
    ax.set_xticks(x + w * (len(names) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_title("Detection quality on labeled sliding windows")
    ax.legend()
    for i in range(len(names)):
        for j, v in enumerate(vals[i]):
            ax.text(x[j] + i * w, v + 0.02, f"{v:.2f}", ha="center",
                    fontsize=7, rotation=90)
    fig.tight_layout()
    out = config.FIG_DIR / "detection_quality.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    # feature importance
    imp = det.get("ml_feature_importance", {})
    if imp:
        fig, ax = plt.subplots(figsize=(9, 5))
        items = sorted(imp.items(), key=lambda kv: kv[1])
        ax.barh([k for k, _ in items], [v for _, v in items], color="#4477aa")
        ax.set_title("Random-forest detector: window-feature importance")
        fig.tight_layout()
        out2 = config.FIG_DIR / "feature_importance.png"
        fig.savefig(out2, bbox_inches="tight")
        plt.close(fig)
        return out, out2
    return out, None


def fig_utility(summary: dict):
    ben, atk = summary["benign_utility"], summary["attacker_experience"]
    labels = ["score MAE", "throttle rate", "fraction L2+", "fraction L3"]
    bv = [ben["score_mae"], ben["throttle_rate"], 0.0,
          0.0]  # benign never reaches L2/L3
    av = [atk["score_mae_vs_raw"], atk["throttle_rate"],
          atk["fraction_level2plus"], atk["fraction_level3"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, bv, 0.4, label="benign partners", color="#2a9d8f")
    ax.bar(x + 0.2, av, 0.4, label="attackers", color="#e76f51")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Defense impact: benign partners vs. attackers")
    ax.legend()
    for xi, v in zip(x - 0.2, bv):
        ax.text(xi, v + 0.01, f"{v:.4f}", ha="center", fontsize=8)
    for xi, v in zip(x + 0.2, av):
        ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    out = config.FIG_DIR / "utility_split.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_evasion(eva: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.35
    attacks = eva["attack"].unique()
    x = np.arange(len(attacks))
    for i, mode in enumerate(["undefended", "defended"]):
        sub = (eva[eva["mode"] == mode].groupby("attack")
               ["bypass_rate_vs_victim"].mean().reindex(attacks))
        ax.bar(x + (i - 0.5) * width, sub.values, width, label=mode)
    ax.set_xticks(x); ax.set_xticklabels(attacks, rotation=15)
    ax.set_ylabel("mean bypass rate vs victim")
    ax.set_title("Surrogate-guided evasion: does the stolen model still find bypasses?")
    ax.legend()
    fig.tight_layout()
    out = config.FIG_DIR / "evasion_bypass.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    fid, eva, summary = _load()
    outs = [fig_fidelity_vs_budget(fid), fig_detection(summary),
            fig_utility(summary), fig_evasion(eva)]
    flat = [p for tup in outs for p in (tup if isinstance(tup, tuple) else (tup,)) if p]
    print("figures written:")
    for p in flat:
        print(" -", p)


if __name__ == "__main__":
    main()
