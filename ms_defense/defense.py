"""Adaptive output defenses applied *after* detection.

Design constraints from the problem statement:
  * trusted, high-volume benign partners must keep essentially exact scores;
  * the perturbation must not be removable by repeated query averaging — hence
    deterministic, seeded from (account, feature vector, rotating salt), so
    the same query always returns the same answer within a salt epoch;
  * degradation must scale with evidence of extraction, not with raw volume.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import config


def _stable_unit_normal(seed_bytes: bytes) -> float:
    """Deterministic standard-normal draw from arbitrary seed bytes."""
    h = hashlib.blake2b(seed_bytes, digest_size=16).digest()
    u1 = (int.from_bytes(h[:8], "little") + 1) / (1 << 64)
    u2 = (int.from_bytes(h[8:], "little") + 1) / (1 << 64)
    return float(np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2))


class AdaptiveDefense:
    def __init__(self, rotate_every: int = config.SALT_ROTATION_QUERIES):
        self.rotate_every = rotate_every
        self._call_count = 0
        self._salt_epoch = 0
        # account-specific salts stop an attacker from averaging noise away
        # by replaying the same vector from different accounts
        self._account_salt: dict[str, int] = {}

    def _salt_for(self, account_id: str) -> int:
        if account_id not in self._account_salt:
            h = hashlib.blake2b(account_id.encode(), digest_size=8).digest()
            self._account_salt[account_id] = int.from_bytes(h, "little")
        return self._account_salt[account_id] ^ self._salt_epoch

    def transform(self, account_id: str, x: np.ndarray, raw_score: float,
                  suspicion: float, level: int) -> float:
        self._call_count += 1
        if self._call_count % self.rotate_every == 0:
            self._salt_epoch += 1

        if level == 0:
            # trusted traffic: no degradation at all
            return raw_score

        quant = config.QUANT_GRID[level]
        sd = config.NOISE_SD[level]
        salt = self._salt_for(account_id)

        # deterministic noise keyed on the actual query: replay-averaging fails
        key = f"{salt}|".encode() + np.round(x, 6).tobytes()
        z = _stable_unit_normal(key)
        # bias noise toward the decision threshold on risky-account responses:
        # this specifically damages the surrogate's boundary geometry while
        # moving trusted labels rarely
        pull = 0.0
        if level >= 2:
            pull = -np.sign(raw_score - config.RISK_THRESHOLD) * sd * 0.8
        noisy = raw_score + sd * z + pull

        # boundary-targeted, deterministic label poisoning. Only L2/L3 queries
        # already within POISON_BAND of the threshold are eligible; the flip
        # decision is a stable hash of the query (replays get the same answer),
        # so the attacker cannot average it away and the API keeps consistent
        # semantics. A flipped score is pushed past the threshold by a margin.
        poison_p = config.LABEL_POISON_P[level]
        if poison_p > 0 and abs(raw_score - config.RISK_THRESHOLD) <= config.POISON_BAND:
            u = 0.5 + 0.5 * _stable_unit_normal(b"flip|" + key)  # deterministic ~(0,1)
            if u < poison_p:
                margin = config.POISON_MARGIN[level]
                if raw_score >= config.RISK_THRESHOLD:
                    noisy = config.RISK_THRESHOLD - margin
                else:
                    noisy = config.RISK_THRESHOLD + margin

        if quant > 0:
            noisy = np.round(noisy / quant) * quant
        return float(np.clip(noisy, 0.0, 1.0))


class NullDefense:
    """Baseline: return raw scores (for attack-fidelity measurement)."""

    def transform(self, account_id, x, raw_score, suspicion, level) -> float:
        return raw_score
