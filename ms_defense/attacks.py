"""Model-stealing attacks and evasion-sample generation.

All attacks talk to the gateway through a single oracle callable:
    oracle(account_id, x, ts) -> {"score", "label", ...}
so they work identically against the raw model and the defended API.

Attack catalogue
----------------
UniformRandom     : blind uniform probing of the feature box
GaussianMimicry   : probes drawn from the benign marginal distribution
                    (attacker possesses / estimates real business data)
BoundarySearch    : active adaptive attack — line search + bisection that
                    concentrates queries on the decision boundary
LabelOnlyBoundary : same, using the binary label only (worst-case attacker
                    when scores are heavily quantized)
DistributedAttack : splits any of the above across N accounts and mixes
                    genuine benign traffic to dilute per-account signals
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class QueryRecord:
    account_id: str
    ts: float
    x: np.ndarray
    score: float
    label: int
    origin: str  # "attack" | "benign_mix"


def _records_to_frame(records: list[QueryRecord]) -> pd.DataFrame:
    X = np.vstack([r.x for r in records])
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["account_id"] = [r.account_id for r in records]
    df["ts"] = [r.ts for r in records]
    df["score"] = [r.score for r in records]
    df["label"] = [r.label for r in records]
    df["origin"] = [r.origin for r in records]
    return df


class BaseAttack:
    def __init__(self, oracle, seed: int = 0):
        self.oracle = oracle
        self.rng = np.random.default_rng(seed)
        self.records: list[QueryRecord] = []

    def _ask(self, account_id, x, ts, origin="attack") -> dict:
        resp = self.oracle(account_id, np.asarray(x, dtype=float), ts)
        self.records.append(QueryRecord(
            account_id, ts, np.asarray(x, dtype=float).ravel().copy(),
            float(resp["score"]), int(resp["label"]), origin,
        ))
        return resp

    def frame(self) -> pd.DataFrame:
        return _records_to_frame(self.records)


class UniformRandomAttack(BaseAttack):
    def run(self, account_id: str, n: int, ts_start: float, gap: float,
            low: np.ndarray, high: np.ndarray) -> pd.DataFrame:
        d = len(low)
        for i in range(n):
            x = self.rng.uniform(low, high)
            self._ask(account_id, x, ts_start + i * gap)
        return self.frame()


class GaussianMimicryAttack(BaseAttack):
    """Probes drawn N(mu, Sigma) fitted on a small pool of real business rows
    the attacker legitimately has access to as a partner."""

    def run(self, account_id: str, n: int, ts_start: float, gap: float,
            benign_pool: np.ndarray, jitter: float = 0.15) -> pd.DataFrame:
        mu = benign_pool.mean(axis=0)
        sd = benign_pool.std(axis=0) + 1e-9
        d = benign_pool.shape[1]
        # mild covariance structure via PCA-free Gaussian on standardized data
        corr = np.corrcoef(benign_pool.T)
        corr = np.eye(d) * (1 - jitter) + corr * jitter
        L = np.linalg.cholesky(corr + np.eye(d) * 1e-8)
        for i in range(n):
            z = self.rng.standard_normal(d)
            x = mu + sd * (L @ z)
            self._ask(account_id, x, ts_start + i * gap)
        return self.frame()


class BoundarySearchAttack(BaseAttack):
    """Active adaptive extraction: bracket and bisect the decision boundary
    along random directions, then sample densely around boundary points.

    use_label_only=True ignores the returned score, making the attack robust
    to score quantization/noise (label-only adversary).
    """

    def __init__(self, oracle, seed: int = 0, use_label_only: bool = False,
                 bisection_iters: int = 12):
        super().__init__(oracle, seed)
        self.use_label_only = use_label_only
        self.bisection_iters = bisection_iters

    def _resp(self, account_id, x, ts):
        r = self._ask(account_id, x, ts)
        return r["label"], r["score"]

    def run(self, account_id: str, n_budget: int, ts_start: float, gap: float,
            low: np.ndarray, high: np.ndarray) -> pd.DataFrame:
        d = len(low)
        ts = ts_start
        center = (low + high) / 2
        half = (high - low) / 2
        budget = n_budget

        def tick():
            nonlocal ts
            t = ts
            ts += gap
            return t

        while len(self.records) < budget:
            x0 = self.rng.uniform(low, high)
            direction = self.rng.standard_normal(d)
            direction /= np.linalg.norm(direction) + 1e-12
            # expanding bracket: find a point with opposite label
            lab0, _ = self._resp(account_id, x0, tick())
            step = half.mean()
            x1 = x0 + step * direction
            found = None
            for _ in range(8):
                if len(self.records) >= budget:
                    break
                lab1, _ = self._resp(account_id, np.clip(x1, low, high), tick())
                if lab1 != lab0:
                    found = (x0, np.clip(x1, low, high), lab0, lab1)
                    break
                step *= 2
                x1 = x0 + step * direction
            if found is None:
                continue
            lo, hi, lab_lo, lab_hi = found
            # bisection to the boundary
            for _ in range(self.bisection_iters):
                if len(self.records) >= budget:
                    break
                mid = (lo + hi) / 2
                lab_mid, score_mid = self._resp(account_id, mid, tick())
                if not self.use_label_only:
                    # dense sampling near the current boundary estimate
                    for k in range(2):
                        if len(self.records) >= budget:
                            break
                        jitter = self.rng.normal(0, 0.05 * half, d)
                        self._resp(account_id, mid + jitter, tick())
                if lab_mid == lab_lo:
                    lo = mid
                else:
                    hi = mid
            # exploit: small random walks around the boundary point
            anchor = (lo + hi) / 2
            for _ in range(4):
                if len(self.records) >= budget:
                    break
                x = anchor + self.rng.normal(0, 0.08 * half, d)
                self._resp(account_id, np.clip(x, low, high), tick())
        return self.frame().head(n_budget)


class DistributedAttack:
    """Orchestrator: spread one attack across many accounts, each looking like
    a modest-volume partner, and interleave genuine benign queries."""

    def __init__(self, oracle, attack_cls, n_accounts: int, seed: int = 0,
                 benign_pool: np.ndarray | None = None, **attack_kwargs):
        self.oracle = oracle
        self.attack_cls = attack_cls
        self.n_accounts = n_accounts
        self.rng = np.random.default_rng(seed)
        self.benign_pool = benign_pool
        self.attack_kwargs = attack_kwargs
        self.records: list[QueryRecord] = []

    def run(self, total_budget: int, ts_start: float, gap: float,
            low: np.ndarray, high: np.ndarray,
            benign_fraction: float = 0.4) -> pd.DataFrame:
        accounts = [f"attacker_acct_{i:02d}" for i in range(self.n_accounts)]
        # round-robin assignment keeps per-account volume low
        per_account = total_budget // self.n_accounts
        all_records: list[QueryRecord] = []
        ts = ts_start
        for ai, acct in enumerate(accounts):
            attack = self.attack_cls(self.oracle, seed=100 + ai,
                                     **self.attack_kwargs)
            n_attack = int(per_account * (1 - benign_fraction))
            if self.attack_cls is GaussianMimicryAttack:
                attack.run(account_id=acct, n=n_attack, ts_start=ts, gap=gap,
                           benign_pool=self.benign_pool)
            else:
                attack.run(account_id=acct, n_budget=n_attack, ts_start=ts,
                           gap=gap, low=low, high=high)
            ts += n_attack * gap
            # mixed-in genuine-looking business queries on the same account
            n_benign = per_account - n_attack
            if n_benign > 0 and self.benign_pool is not None:
                for j in range(n_benign):
                    x = self.benign_pool[self.rng.integers(len(self.benign_pool))]
                    r = attack._ask(acct, x, ts, origin="benign_mix")
                    ts += gap
            all_records.extend(attack.records)
        return _records_to_frame(all_records)


def generate_evasion_samples(oracle, X_risky: np.ndarray, n_steps: int = 60,
                             seed: int = 0, target: float = 0.45) -> tuple[np.ndarray, dict]:
    """Gradient-free evasion: push genuinely-risky samples below the API's
    decision threshold with the smallest L2 perturbation the surrogate-guided
    attacker can find. Coordinate descent on the *live* API.

    Returns the perturbed samples and summary stats.
    """
    rng = np.random.default_rng(seed)
    n, d = X_risky.shape
    X_adv = X_risky.copy()
    l2_total = []
    successes = 0
    for i in range(n):
        x = X_risky[i].copy()
        x0 = x.copy()
        base_step = 0.15
        for step in range(n_steps):
            coord = rng.integers(d)
            delta = rng.choice([-1, 1]) * base_step * (1 - step / n_steps)
            cand = x.copy()
            cand[coord] += delta
            resp = oracle("evasion_attacker", cand, float(step))
            if resp["score"] < target:
                x = cand
                break
            # accept moves that reduce the score (greedy)
            cur = oracle("evasion_attacker", x, float(step) + 0.5)
            if resp["score"] < cur["score"]:
                x = cand
        final = oracle("evasion_attacker", x, float(n_steps))
        if final["label"] == 0:
            successes += 1
        l2_total.append(float(np.linalg.norm(x - x0)))
        X_adv[i] = x
    stats = {
        "bypass_rate": successes / n,
        "mean_l2": float(np.mean(l2_total)),
        "median_l2": float(np.median(l2_total)),
    }
    return X_adv, stats
