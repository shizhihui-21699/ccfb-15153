"""Ablation: measure how much each defense mechanism contributes to raising
the attacker's copy cost, while tracking benign-partner impact.

Variants (all driven by the same detector):
  null              : raw scores, detection only (control)
  noise_only        : deterministic Gaussian noise, no quantization/poisoning
  quant_only        : score quantization only
  noise_quant       : noise + quantization (no label poisoning)
  full              : noise + quantization + boundary label poisoning (+throttle)
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ms_defense import config
from ms_defense.data import load_dataset
from ms_defense.victim import load_model
from ms_defense.defense import AdaptiveDefense, NullDefense
from ms_defense.detection import RuleBasedDetector
from ms_defense.experiment import (
    build_context, make_gateway, run_benign_traffic,
    run_single_account_attacks, evaluate_stolen_set,
)


@dataclass
class VariantDefense:
    use_noise: bool = False
    use_quant: bool = False
    use_poison: bool = False
    name: str = "custom"

    def transform(self, account_id, x, raw_score, suspicion, level):
        if level == 0:
            return raw_score
        sd = config.NOISE_SD[level] if self.use_noise else 0.0
        quant = config.QUANT_GRID[level] if self.use_quant else 0.0
        out = raw_score
        if sd > 0:
            from ms_defense.defense import _stable_unit_normal
            salt = 0  # variant-local deterministic salt
            key = f"{salt}|{account_id}|".encode() + np.round(x, 6).tobytes()
            out += sd * _stable_unit_normal(key)
            if level >= 2:
                out -= np.sign(raw_score - 0.5) * sd * 0.8
        if self.use_poison and level >= 2 and abs(raw_score - 0.5) <= config.POISON_BAND:
            from ms_defense.defense import _stable_unit_normal
            key = f"0|{account_id}|".encode() + np.round(x, 6).tobytes()
            u = 0.5 + 0.5 * _stable_unit_normal(b"flip|" + key)
            p = config.LABEL_POISON_P[level]
            if u < p:
                margin = config.POISON_MARGIN[level]
                out = 0.5 - margin if raw_score >= 0.5 else 0.5 + margin
        if quant > 0:
            out = np.round(out / quant) * quant
        return float(np.clip(out, 0.0, 1.0))


VARIANTS = {
    "null": lambda: NullDefense(),
    "noise_only": lambda: VariantDefense(use_noise=True, name="noise_only"),
    "quant_only": lambda: VariantDefense(use_quant=True, name="quant_only"),
    "noise_quant": lambda: VariantDefense(use_noise=True, use_quant=True, name="noise_quant"),
    "full": lambda: AdaptiveDefense(),
}


def main() -> None:
    ds = load_dataset()
    victim, _, _ = load_model()
    ctx = build_context(ds, victim)
    budget = 2_000
    rows = []
    for vname, factory in VARIANTS.items():
        det = RuleBasedDetector(ctx.extractor)
        gw = make_gateway(ctx, det, factory())
        run_benign_traffic(ctx, gw, n_partners=4, queries_each=800)
        attacks = run_single_account_attacks(ctx, gw, budget=budget, t0=3_000_000.0)
        audit = gw.audit.to_frame()
        ben = audit[audit.account_id.str.startswith("partner_")]
        for atk_name, frame in attacks.items():
            fid = evaluate_stolen_set(ctx, frame)
            if not fid:
                continue
            rows.append({
                "variant": vname, "attack": atk_name,
                "label_agreement": fid["label_agreement"],
                "score_rmse": fid["score_rmse"],
                "surrogate_auc": fid["surrogate_auc"],
                "bypass_rate": fid["evasion_bypass_rate_vs_victim"],
                "benign_mae": float(np.abs(ben.returned_score - ben.raw_score).mean()),
                "benign_label_flips": float((ben.raw_label != ben.returned_label).mean()),
                "attack_frac_l2plus": float((audit[~audit.account_id.str.startswith("partner_")]["level"] >= 2).mean()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(config.REPORT_DIR / "ablation.csv", index=False)
    agg = df.groupby("variant").agg(
        mean_agreement=("label_agreement", "mean"),
        mean_rmse=("score_rmse", "mean"),
        mean_surrogate_auc=("surrogate_auc", "mean"),
        mean_bypass=("bypass_rate", "mean"),
        benign_mae=("benign_mae", "first"),
        benign_flips=("benign_label_flips", "first"),
    ).round(4).reindex(VARIANTS.keys())
    print(agg.to_string())
    (config.REPORT_DIR / "ablation_summary.json").write_text(
        json.dumps(agg.reset_index().to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
