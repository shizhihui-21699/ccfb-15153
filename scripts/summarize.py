"""Print a concise, decision-oriented summary of the experiment artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ms_defense import config


def main() -> None:
    summary = json.loads((config.REPORT_DIR / "run_summary.json").read_text())
    fid = pd.read_csv(config.REPORT_DIR / "fidelity_curves.csv")

    print("# 1. DETECTION (window-level)")
    for name, m in summary["detection"].items():
        if name == "ml_feature_importance":
            continue
        print(f"  {name:16s} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} FPR={m['fpr']:.4f} AUC={m['roc_auc']:.3f}")

    print("\n# 2. BENIGN PARTNER IMPACT (the hard constraint)")
    b = summary["benign_utility"]
    print(f"  queries={b['n_benign_queries']}  score MAE={b['score_mae']:.2e}  "
          f"label flips={b['label_flip_rate']:.4f}  throttle={b['throttle_rate']:.4f}  "
          f"max level={b['max_level_reached']}")

    print("\n# 3. ATTACKER EXPERIENCE")
    a = summary["attacker_experience"]
    print(f"  queries={a['n_attack_queries']}  throttle={a['throttle_rate']:.2%}  "
          f"L2+={a['fraction_level2plus']:.2%}  L3={a['fraction_level3']:.2%}  "
          f"score MAE vs raw={a['score_mae_vs_raw']:.4f}")

    print("\n# 4. SURROGATE FIDELITY (undefended -> defended), by budget")
    piv = fid.pivot_table(index=["attack", "budget"], columns="mode",
                          values=["label_agreement", "surrogate_auc", "score_rmse"])
    for atk in fid["attack"].unique():
        print(f"\n  [{atk}]")
        sub = fid[fid.attack == atk].sort_values("budget")
        for _, r in sub.iterrows():
            print(f"   n={int(r.budget):5d} {r['mode']:10s} "
                  f"agree={r.label_agreement:.3f} auc={r.surrogate_auc:.3f} "
                  f"rmse={r.score_rmse:.3f} bypass={r.evasion_bypass_rate_vs_victim:.3f}")

    print("\n# 5. CROSS-ACCOUNT COORDINATION")
    ca = summary["cross_account"]
    print(f"  attacker accounts: {len(ca['attacker_accounts'])} | "
          f"coordinated pairs: {len(ca['correlated_pairs'])}")
    for p in ca["correlated_pairs"][:5]:
        print(f"   {p['a']} <-> {p['b']}  jaccard/score={p['jaccard']}")

    abl = config.REPORT_DIR / "ablation.csv"
    if abl.exists():
        print("\n# 6. ABLATION (mean over attacks, budget=2000)")
        df = pd.read_csv(abl)
        agg = df.groupby("variant").agg(
            agreement=("label_agreement", "mean"),
            rmse=("score_rmse", "mean"),
            surrogate_auc=("surrogate_auc", "mean"),
            bypass=("bypass_rate", "mean"),
            benign_mae=("benign_mae", "first"),
        ).round(4)
        print(agg.to_string())


if __name__ == "__main__":
    main()
