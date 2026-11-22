"""Experiment orchestration: build mixed traffic, run all attacks against the
undefended and defended API, and assemble the result tables."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ms_defense import config
from ms_defense.api import ScoringGateway
from ms_defense.attacks import (
    UniformRandomAttack, GaussianMimicryAttack, BoundarySearchAttack,
    DistributedAttack,
)
from ms_defense.traffic import BenignPartner, build_benign_pool
from ms_defense.detection import RuleBasedDetector, MLDetector
from ms_defense.detection_features import WindowFeatureExtractor
from ms_defense.defense import AdaptiveDefense, NullDefense
from ms_defense.evaluation import train_surrogate, fidelity, evasion_bypass


@dataclass
class ExperimentContext:
    ds: object
    victim: object
    benign_pool: np.ndarray
    low: np.ndarray
    high: np.ndarray

    @property
    def extractor(self) -> WindowFeatureExtractor:
        return WindowFeatureExtractor(self.benign_pool)


def build_context(ds, victim) -> ExperimentContext:
    pool = build_benign_pool(ds)
    # attacker-known feature box estimated from a small legitimate sample
    low = np.quantile(pool, 0.005, axis=0)
    high = np.quantile(pool, 0.995, axis=0)
    return ExperimentContext(ds, victim, pool, low, high)


def make_gateway(ctx: ExperimentContext, detector, defense) -> ScoringGateway:
    return ScoringGateway(ctx.victim, detector, defense)


def run_benign_traffic(ctx: ExperimentContext, gateway: ScoringGateway,
                       n_partners: int = config.N_BENIGN_PARTNERS,
                       queries_each: int = config.BENIGN_QUERIES_PER_PARTNER,
                       t0: float = 0.0) -> pd.DataFrame:
    frames = []
    for p in range(n_partners):
        partner = BenignPartner(gateway.score, seed=1_000 + p)
        acct = f"partner_{p:02d}"
        frames.append(partner.run(
            acct, queries_each, t0 + p * 10_000, ctx.benign_pool,
            rate_qps=float(6 + (p % 4)),
        ))
    return pd.concat(frames, ignore_index=True)


def run_single_account_attacks(ctx: ExperimentContext, gateway: ScoringGateway,
                               budget: int, t0: float = 1_000_000.0) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    gap = 0.05  # 20 qps — machine cadence

    atk = UniformRandomAttack(gateway.score, seed=11)
    out["uniform_random"] = atk.run("atk_uniform", budget, t0, gap, ctx.low, ctx.high)

    atk = GaussianMimicryAttack(gateway.score, seed=22)
    out["gaussian_mimicry"] = atk.run("atk_mimic", budget, t0 + budget, gap,
                                      ctx.benign_pool)

    atk = BoundarySearchAttack(gateway.score, seed=33)
    out["boundary_search"] = atk.run("atk_boundary", budget, t0 + 2 * budget,
                                     gap, ctx.low, ctx.high)

    atk = BoundarySearchAttack(gateway.score, seed=44, use_label_only=True)
    out["label_only_boundary"] = atk.run("atk_labelonly", budget,
                                         t0 + 3 * budget, gap, ctx.low, ctx.high)
    return out


def run_distributed_attack(ctx: ExperimentContext, gateway: ScoringGateway,
                           total_budget: int, t0: float = 5_000_000.0) -> pd.DataFrame:
    atk = DistributedAttack(
        gateway.score, BoundarySearchAttack,
        n_accounts=config.N_ATTACK_ACCOUNTS, seed=55,
        benign_pool=ctx.benign_pool,
    )
    return atk.run(total_budget, t0, gap=0.05, low=ctx.low, high=ctx.high,
                   benign_fraction=0.4)


# --------------------------------------------------------------------------
# Surrogate / fidelity / evasion evaluation for one stolen query set
# --------------------------------------------------------------------------
def evaluate_stolen_set(ctx: ExperimentContext, stolen: pd.DataFrame,
                        surrogate_kind: str = "gbt") -> dict:
    attack_rows = stolen[stolen["origin"] == "attack"]
    X = attack_rows[[c for c in stolen.columns if c.startswith("f")]].to_numpy()
    scores = attack_rows["score"].to_numpy()
    if len(X) < 50:
        return {}
    surrogate = train_surrogate(X, scores, kind=surrogate_kind)
    fid = fidelity(surrogate, ctx.victim, ctx.ds.X_test, ctx.ds.y_test)
    risky_idx = np.where(ctx.ds.y_test == 1)[0][:60]
    eva = evasion_bypass(surrogate, ctx.victim, ctx.ds.X_test[risky_idx], seed=3)
    fid.update({f"evasion_{k}": v for k, v in eva.items()})
    fid["n_queries"] = len(X)
    return fid
