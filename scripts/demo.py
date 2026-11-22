"""Fast end-to-end demo (~1 minute): one attack budget, full defense loop.

Shows the complete chain on a small scale:
  victim model -> benign + attack traffic -> detection levels ->
  adaptive score rewriting -> surrogate fidelity -> evasion transfer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ms_defense.data import load_dataset
from ms_defense.victim import load_model, train_victim, save_model
from ms_defense.experiment import (
    build_context, make_gateway, run_benign_traffic,
    run_single_account_attacks, evaluate_stolen_set,
)
from ms_defense.detection import RuleBasedDetector
from ms_defense.defense import NullDefense, AdaptiveDefense


def _ensure_victim(ds):
    try:
        m, _, _ = load_model()
        return m
    except FileNotFoundError:
        m = train_victim(ds)
        save_model(m, ds)
        return m


def main() -> None:
    ds = load_dataset()
    victim = _ensure_victim(ds)
    ctx = build_context(ds, victim)
    budget = 800

    for tag, defense in [("UNDEFENDED", NullDefense()), ("DEFENDED", AdaptiveDefense())]:
        print(f"\n{'='*70}\n{tag} API\n{'='*70}")
        det = RuleBasedDetector(ctx.extractor)
        gw = make_gateway(ctx, det, defense)
        run_benign_traffic(ctx, gw, n_partners=3, queries_each=500)
        attacks = run_single_account_attacks(ctx, gw, budget=budget, t0=2_000_000.0)
        audit = gw.audit.to_frame()
        ben = audit[audit.account_id.str.startswith("partner_")]
        atk = audit[~audit.account_id.str.startswith("partner_")]

        print(f"benign: {len(ben)} queries | score MAE="
              f"{np.abs(ben.returned_score-ben.raw_score).mean():.2e} | "
              f"label flips={(ben.raw_label!=ben.returned_label).mean():.3f} | "
              f"max level={int(ben.level.max())}")
        print(f"attack: {len(atk)} queries | level2+="
              f"{(atk.level>=2).mean():.2%} | level3={(atk.level==3).mean():.2%} | "
              f"throttled={atk.throttled.mean():.2%}")
        print("\nsurrogate extraction quality by attack:")
        for name, frame in attacks.items():
            fid = evaluate_stolen_set(ctx, frame)
            if fid:
                print(f"  {name:20s} agreement={fid['label_agreement']:.3f} "
                      f"rmse={fid['score_rmse']:.3f} auc={fid['surrogate_auc']:.3f} "
                      f"bypass_vs_victim={fid['evasion_bypass_rate_vs_victim']:.3f}")
        if tag == "DEFENDED":
            pairs = det.correlated_accounts()
            print(f"\ncross-account coordinated pairs detected: {len(pairs)}")
            for a, b, s in pairs[:4]:
                print(f"  {a} <-> {b}  coordination={s:.3f}")


if __name__ == "__main__":
    main()
