"""Benign partner traffic and traffic-mixing helpers."""
from __future__ import annotations

import numpy as np

from .attacks import BaseAttack


class BenignPartner(BaseAttack):
    """Realistic high-volume partner traffic.

    Characteristics of genuine business load (used both as the negative class
    for detector evaluation and as the attacker's camouflage pool):
      - rows resampled from the real feature distribution (operational data),
      - temporal bursts: Poisson arrivals with partner-specific rate,
      - low feature novelty: queries cluster in a narrow operational manifold,
      - no systematic targeting of the decision boundary or feature tails.
    """

    def run(self, account_id: str, n: int, ts_start: float,
            benign_pool: np.ndarray, rate_qps: float = 8.0,
            burstiness: float = 1.5) -> "pd.DataFrame":
        rng = self.rng
        ts = ts_start
        # Poisson gaps with over-dispersion (bursts + quiet stretches)
        mean_gap = 1.0 / rate_qps
        for i in range(n):
            idx = rng.integers(len(benign_pool))
            x = benign_pool[idx].copy()
            # small measurement noise, same scale as real data jitter
            x += rng.normal(0, 0.03, size=x.shape)
            self._ask(account_id, x, ts, origin="benign_mix")
            gap = rng.exponential(mean_gap) * rng.gamma(burstiness, 1 / burstiness)
            ts += max(gap, 1e-4)
        return self.frame()


def build_benign_pool(ds, n: int = 1_200, seed: int = 7) -> np.ndarray:
    """Operational data the partner legitimately sees: draws from the
    underlying distribution (train split) with slight resampling noise."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ds.X_train), size=n)
    pool = ds.X_train[idx].copy()
    pool += rng.normal(0, 0.05, size=pool.shape)
    return pool
