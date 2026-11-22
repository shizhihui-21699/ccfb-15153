"""API gateway: the only component the partner-facing scoring endpoint exposes.

Every call is audit-logged. Detection consumes the same audit stream a real
service would (account id, timestamp, feature vector, returned score), and the
defense policy rewrites the score *after* detection has scored the account —
so benign partners on trusted accounts see essentially untouched scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


@dataclass
class AuditEntry:
    seq: int
    account_id: str
    ts: float
    x: np.ndarray
    raw_score: float
    returned_score: float
    raw_label: int
    returned_label: int
    suspicion: float
    level: int
    throttled: bool
    quant_step: float
    noise_sd: float


class AuditLog:
    """Columnar audit store; mirrors what would land in Kafka/ClickHouse."""

    def __init__(self):
        self._rows: list[AuditEntry] = []

    def append(self, entry: AuditEntry) -> None:
        self._rows.append(entry)

    def to_frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame()
        rows = []
        for e in self._rows:
            d = e.__dict__.copy()
            d.pop("x")
            rows.append(d)
        df = pd.DataFrame(rows)
        return df

    def feature_matrix(self) -> np.ndarray:
        if not self._rows:
            return np.empty((0, 0))
        return np.vstack([e.x for e in self._rows])

    def __len__(self) -> int:
        return len(self._rows)


class ScoringGateway:
    """Partner-facing scoring API with detection + adaptive defense hooks."""

    def __init__(self, model, detector, defense, threshold: float = config.RISK_THRESHOLD):
        self.model = model
        self.detector = detector
        self.defense = defense
        self.threshold = threshold
        self.audit = AuditLog()
        self._seq = 0
        # simulated token buckets per account: account -> (tokens, last_ts)
        self._buckets: dict[str, tuple[float, float]] = {}

    def _allow(self, account_id: str, ts: float, qps: float | None) -> bool:
        if qps is None:
            return True
        capacity = max(qps, 1.0)
        tokens, last = self._buckets.get(account_id, (capacity, ts))
        tokens = min(capacity, tokens + (ts - last) * qps)
        if tokens < 1.0:
            self._buckets[account_id] = (tokens, ts)
            return False
        self._buckets[account_id] = (tokens - 1.0, ts)
        return True

    def score(self, account_id: str, x: np.ndarray, ts: float) -> dict:
        """Score one feature vector. x may be 1-D or a single-row 2-D array."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        raw = float(self.model.predict_proba(x)[0, 1])
        raw_label = int(raw >= self.threshold)

        # 1) detection sees the request before any output manipulation
        suspicion, level = self.detector.observe(account_id, ts, x[0], raw)
        qps = config.RATE_LIMIT_QPS[level]
        throttled = not self._allow(account_id, ts, qps)

        # 2) defense rewrites the output according to current suspicion
        returned = self.defense.transform(account_id, x[0], raw, suspicion, level)
        returned = float(np.clip(returned, 0.0, 1.0))
        returned_label = int(returned >= self.threshold)

        entry = AuditEntry(
            seq=self._seq, account_id=account_id, ts=ts, x=x[0],
            raw_score=raw, returned_score=returned,
            raw_label=raw_label, returned_label=returned_label,
            suspicion=suspicion, level=level, throttled=throttled,
            quant_step=config.QUANT_GRID[level], noise_sd=config.NOISE_SD[level],
        )
        self.audit.append(entry)
        self._seq += 1
        return {
            "score": returned,
            "label": returned_label,
            "throttled": throttled,
            "suspicion_level": level,
        }

    def score_batch(self, account_id: str, X: np.ndarray, ts_start: float,
                    gap: float = 1.0) -> list[dict]:
        out = []
        for i, row in enumerate(np.atleast_2d(X)):
            out.append(self.score(account_id, row, ts_start + i * gap))
        return out
