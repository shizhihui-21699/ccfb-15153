"""End-to-end runner: attacks vs. undefended/defended API, detection quality,
surrogate fidelity degradation, evasion transfer, and benign-partner utility.

Outputs (artifacts/reports/):
  detection_metrics.json   — precision/recall/FPR/AUC, feature importance
  fidelity_curves.csv      — surrogate fidelity vs. query budget, both modes
  evasion_summary.csv      — bypass-rate transfer to the victim
  benign_utility.json      — score MAE / label flips / throttle rate
  cross_account.json       — distributed-attack correlation findings
  run_summary.json         — everything in one place
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ms_defense import config
from ms_defense.data import load_dataset
from ms_defense.victim import load_model, train_victim, save_model, evaluate
from ms_defense.defense import NullDefense, AdaptiveDefense
from ms_defense.detection import RuleBasedDetector
from ms_defense.experiment import (
    build_context, make_gateway, run_benign_traffic,
    run_single_account_attacks, run_distributed_attack, evaluate_stolen_set,
)
from ms_defense.detection_eval import collect_window_samples, evaluate_detectors


def _ensure_victim(ds):
    try:
        model, _, _ = load_model()
        return model
    except FileNotFoundError:
        model = train_victim(ds)
        save_model(model, ds)
        return model


def run_mode(ctx, defense, budgets, include_benign=True, tag=""):
    detector = RuleBasedDetector(ctx.extractor)
    gateway = make_gateway(ctx, detector, defense)

    benign_frames = []
    if include_benign:
        benign_frames.append(run_benign_traffic(ctx, gateway))

    fidelity_rows, evasion_rows = [], []
    attack_frames_by_budget = {}
    for budget in budgets:
        attacks = run_single_account_attacks(ctx, gateway, budget,
                                             t0=1_000_000 + budget)
        attack_frames_by_budget[budget] = attacks
        for name, frame in attacks.items():
            fid = evaluate_stolen_set(ctx, frame)
            if fid:
                row = {"mode": tag, "attack": name, "budget": budget, **fid}
                fidelity_rows.append(row)
                evasion_rows.append({
                    "mode": tag, "attack": name, "budget": budget,
                    "bypass_rate_vs_victim": fid["evasion_bypass_rate_vs_victim"],
                    "score_drop_on_victim": fid["evasion_score_drop_on_victim"],
                    "label_agreement": fid["label_agreement"],
                    "surrogate_auc": fid["surrogate_auc"],
                })

    distributed = run_distributed_attack(ctx, gateway, total_budget=4_000)
    dist_fid = evaluate_stolen_set(ctx, distributed)
    if dist_fid:
        fidelity_rows.append({"mode": tag, "attack": "distributed_boundary",
                              "budget": 4_000, **dist_fid})
        evasion_rows.append({
            "mode": tag, "attack": "distributed_boundary", "budget": 4_000,
            "bypass_rate_vs_victim": dist_fid["evasion_bypass_rate_vs_victim"],
            "score_drop_on_victim": dist_fid["evasion_score_drop_on_victim"],
            "label_agreement": dist_fid["label_agreement"],
            "surrogate_auc": dist_fid["surrogate_auc"],
        })

    audit = gateway.audit.to_frame()
    return {
        "gateway": gateway,
        "detector": detector,
        "audit": audit,
        "benign_frames": benign_frames,
        "attack_frames": attack_frames_by_budget,
        "distributed": distributed,
        "fidelity": pd.DataFrame(fidelity_rows),
        "evasion": pd.DataFrame(evasion_rows),
    }


def benign_utility(ctx, audit: pd.DataFrame) -> dict:
    """Impact on legitimate partners — the hard constraint."""
    ben = audit[audit["account_id"].str.startswith("partner_")]
    if ben.empty:
        return {}
    return {
        "n_benign_queries": int(len(ben)),
        "score_mae": float(np.mean(np.abs(ben["returned_score"] - ben["raw_score"]))),
        "score_rmse": float(np.sqrt(np.mean(
            (ben["returned_score"] - ben["raw_score"]) ** 2))),
        "label_flip_rate": float(np.mean(ben["raw_label"] != ben["returned_label"])),
        "throttle_rate": float(ben["throttled"].mean()),
        "max_level_reached": int(ben["level"].max()),
        "mean_suspicion": float(ben["suspicion"].mean()),
        "p99_suspicion": float(ben["suspicion"].quantile(0.99)),
    }


def attacker_utility(audit: pd.DataFrame) -> dict:
    atk = audit[~audit["account_id"].str.startswith("partner_")]
    if atk.empty:
        return {}
    return {
        "n_attack_queries": int(len(atk)),
        "throttle_rate": float(atk["throttled"].mean()),
        "fraction_level2plus": float((atk["level"] >= 2).mean()),
        "fraction_level3": float((atk["level"] == 3).mean()),
        "mean_suspicion": float(atk["suspicion"].mean()),
        "score_mae_vs_raw": float(np.mean(
            np.abs(atk["returned_score"] - atk["raw_score"]))),
    }


def main() -> None:
    ds = load_dataset()
    victim = _ensure_victim(ds)
    print("[run] victim metrics:", evaluate(victim, ds))
    ctx = build_context(ds, victim)
    budgets = config.ATTACK_BUDGETS

    print("[run] phase 1: undefended API (attack baseline)")
    undef = run_mode(ctx, NullDefense(), budgets, tag="undefended")
    print("[run] phase 2: defended API (detection + adaptive response)")
    defended = run_mode(ctx, AdaptiveDefense(), budgets, tag="defended")

    # ---- detection evaluation on the defended run's traffic --------------
    print("[run] phase 3: detector evaluation")
    Xw, yw, meta = collect_window_samples(
        ctx.extractor, defended["benign_frames"],
        defended["attack_frames"][budgets[-1]], defended["distributed"],
    )
    det_metrics, ml_detector = evaluate_detectors(Xw, yw, ctx.extractor)

    # ---- cross-account correlation ----------------------------------------
    corr = defended["detector"].correlated_accounts()
    cross_account = {
        "correlated_pairs": [
            {"a": a, "b": b, "jaccard": round(j, 4)} for a, b, j in corr
        ],
        "attacker_accounts": sorted(
            defended["distributed"]["account_id"].unique().tolist()),
    }

    # ---- utility -----------------------------------------------------------
    util_benign = benign_utility(ctx, defended["audit"])
    util_attack = attacker_utility(defended["audit"])

    # ---- persist -----------------------------------------------------------
    fidelity = pd.concat([undef["fidelity"], defended["fidelity"]], ignore_index=True)
    evasion = pd.concat([undef["evasion"], defended["evasion"]], ignore_index=True)
    fidelity.to_csv(config.REPORT_DIR / "fidelity_curves.csv", index=False)
    evasion.to_csv(config.REPORT_DIR / "evasion_summary.csv", index=False)

    summary = {
        "dataset": ds.name,
        "victim": evaluate(victim, ds),
        "detection": det_metrics,
        "benign_utility": util_benign,
        "attacker_experience": util_attack,
        "cross_account": cross_account,
        "n_windows_labeled": {"total": len(yw),
                              "benign": int((yw == 0).sum()),
                              "attack": int((yw == 1).sum())},
    }
    (config.REPORT_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    (config.REPORT_DIR / "detection_metrics.json").write_text(
        json.dumps(det_metrics, indent=2, default=float))
    (config.REPORT_DIR / "benign_utility.json").write_text(
        json.dumps(util_benign, indent=2))
    (config.REPORT_DIR / "cross_account.json").write_text(
        json.dumps(cross_account, indent=2, default=float))
    defended["audit"].to_csv(config.REPORT_DIR / "audit_defended.csv", index=False)
    undef["audit"].to_csv(config.REPORT_DIR / "audit_undefended.csv", index=False)

    print("\n=== detection ===")
    for k, v in det_metrics.items():
        if k != "ml_feature_importance":
            print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                      for kk, vv in v.items()})
    print("\n=== benign utility ===", json.dumps(util_benign, indent=2))
    print("\n=== attacker experience ===", json.dumps(util_attack, indent=2))
    print(f"\n[run] reports written to {config.REPORT_DIR}")


if __name__ == "__main__":
    main()
