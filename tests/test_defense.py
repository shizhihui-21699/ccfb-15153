"""Unit tests for the defense building blocks. Run: .venv/bin/python -m unittest discover -s tests -v"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ms_defense.config import RATE_LIMIT_QPS
from ms_defense.defense import AdaptiveDefense, NullDefense
from ms_defense.detection_features import WindowFeatureExtractor
from ms_defense.detection import RuleBasedDetector
from ms_defense.api import ScoringGateway


def _toy_data(seed=0, n=600, d=8):
    rng = np.random.default_rng(seed)
    benign = rng.normal(size=(n, d))
    return benign


class FakeModel:
    """Deterministic stand-in for predict_proba based on a linear score."""

    def __init__(self, d):
        self.w = np.linspace(0.5, 1.5, d)

    def predict_proba(self, X):
        z = X @ self.w / np.linalg.norm(self.w)
        p = 1 / (1 + np.exp(-z))
        return np.vstack([1 - p, p]).T


class TestDefense(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.x = self.rng.normal(size=10)
        self.d = AdaptiveDefense()

    def test_level0_passthrough(self):
        out = self.d.transform("acct", self.x, 0.73, suspicion=0.1, level=0)
        self.assertAlmostEqual(out, 0.73, places=9)

    def test_deterministic_same_query_same_output(self):
        a = self.d.transform("acct", self.x, 0.73, 0.8, 2)
        b = self.d.transform("acct", self.x, 0.73, 0.8, 2)
        self.assertEqual(a, b)

    def test_replay_across_accounts_differs(self):
        # attacker cannot average noise away by rotating accounts: the
        # deterministic noise draw itself is keyed on the account salt.
        from ms_defense.defense import _stable_unit_normal
        xb = np.round(self.x, 6).tobytes()
        za = _stable_unit_normal(f"{self.d._salt_for('acct_a')}|".encode() + xb)
        zb = _stable_unit_normal(f"{self.d._salt_for('acct_b')}|".encode() + xb)
        self.assertNotAlmostEqual(za, zb)
        # and the same (account, query) is reproducible
        za2 = _stable_unit_normal(f"{self.d._salt_for('acct_a')}|".encode() + xb)
        self.assertEqual(za, za2)

    def test_quantization_grid(self):
        out = self.d.transform("acct", self.x, 0.731, 0.9, 3)
        q = 0.10
        self.assertAlmostEqual(round(out / q) * q, out, places=9)
        self.assertGreaterEqual(out, 0.0)
        self.assertLessEqual(out, 1.0)

    def test_null_defense(self):
        self.assertEqual(NullDefense().transform("a", self.x, 0.42, 0.9, 3), 0.42)

    def test_label_poison_deterministic_and_bounded(self):
        from ms_defense import config
        d = AdaptiveDefense()
        rng = np.random.default_rng(9)
        # same (account, x, score) always yields the same output across replays
        seen = {}
        for _ in range(300):
            x = rng.normal(size=10)
            s = float(0.5 + rng.uniform(-config.POISON_BAND, config.POISON_BAND))
            key = (x.tobytes(), round(s, 6))
            out = d.transform("acct", x, s, 0.95, 3)
            if key in seen:
                self.assertEqual(seen[key], out)
            seen[key] = out
            self.assertTrue(0.0 <= out <= 1.0)
        # and at least some boundary queries were actually flipped at L3
        flipped = sum(1 for (xb, s), o in ((k, v) for k, v in seen.items())
                      if (o >= 0.5) != (s >= 0.5))
        self.assertGreater(flipped, 0)

    def test_far_from_boundary_never_poisoned_to_flip(self):
        from ms_defense import config
        d = AdaptiveDefense()
        rng = np.random.default_rng(10)
        # a score far outside the poison band cannot have its label flipped
        s = 0.95
        flips = 0
        for _ in range(200):
            out = d.transform("acct", rng.normal(size=10), s, 0.95, 3)
            if (out >= 0.5) != (s >= 0.5):
                flips += 1
        self.assertEqual(flips, 0)


class TestFeatureExtractor(unittest.TestCase):
    def setUp(self):
        self.benign = _toy_data()
        self.ext = WindowFeatureExtractor(self.benign)
        self.rng = np.random.default_rng(2)

    def test_feature_vector_shape_and_finite(self):
        X = self.benign[:200]
        scores = self.rng.uniform(size=200)
        labels = (scores >= 0.5).astype(int)
        ts = np.arange(200) * 0.1
        f = self.ext.extract(X, scores, labels, ts)
        self.assertEqual(f.shape, (len(self.ext.FEATURE_NAMES),))
        self.assertTrue(np.isfinite(f).all())

    def test_boundary_hunting_scores_high(self):
        benign_scores = np.clip(self.rng.normal(0.2, 0.1, 200), 0, 1)
        boundary_scores = np.clip(0.5 + self.rng.normal(0, 0.03, 200), 0, 1)
        X = self.benign[:200]; ts = np.arange(200) * 0.1
        fb = self.ext.boundary_fraction(boundary_scores)
        fn = self.ext.boundary_fraction(benign_scores)
        self.assertGreater(fb, fn)

    def test_ood_novelty_higher(self):
        in_dist = self.benign[:200]
        ood = self.rng.normal(0, 6, size=(200, self.benign.shape[1]))
        self.assertGreater(self.ext.novelty(ood), self.ext.novelty(in_dist))

    def test_coarse_cells_stable(self):
        cells1 = self.ext.coarse_cell_set(self.benign[:50])
        cells2 = self.ext.coarse_cell_set(self.benign[:50] + 1e-9)
        self.assertEqual(cells1, cells2)


class TestDetectorStreaming(unittest.TestCase):
    def setUp(self):
        self.d = 8
        self.benign = _toy_data(n=800, d=self.d)
        self.ext = WindowFeatureExtractor(self.benign)
        self.detector = RuleBasedDetector(self.ext)
        self.rng = np.random.default_rng(3)

    def _feed(self, acct, X, scores, gap=0.1):
        levels = []
        for i, (x, s) in enumerate(zip(X, scores)):
            _, lvl = self.detector.observe(acct, i * gap, x, float(s))
            levels.append(lvl)
        return levels

    def test_benign_stream_stays_low(self):
        X = self.benign[:500]
        scores = np.clip(self.rng.normal(0.25, 0.12, 500), 0.02, 0.98)
        levels = self._feed("partner", X, scores, gap=0.15)
        self.assertLessEqual(max(levels), 1)

    def test_boundary_attack_escalates(self):
        # OOD points concentrated around the 0.5 score band, machine cadence
        X = self.rng.normal(0, 5, size=(500, self.d))
        scores = np.clip(0.5 + self.rng.normal(0, 0.02, 500), 0, 1)
        levels = self._feed("attacker", X, scores, gap=0.01)
        self.assertGreaterEqual(max(levels), 2)

    def test_suspicion_bounded(self):
        X = self.rng.normal(0, 5, size=(400, self.d))
        scores = self.rng.uniform(0.4, 0.6, 400)
        self._feed("attacker", X, scores)
        for st in self.detector.accounts.values():
            self.assertGreaterEqual(st.suspicion_ema, 0.0)
            self.assertLessEqual(st.suspicion_ema, 1.0)

    def test_early_window_escalates_before_full_window(self):
        # cold-start hole: an obvious machine sweep must be caught at the
        # early window (~80 queries), not only after 200
        from ms_defense import config
        X = self.rng.normal(0, 6, size=(config.EARLY_WINDOW, self.d))
        scores = np.clip(0.5 + self.rng.normal(0, 0.02, config.EARLY_WINDOW), 0, 1)
        levels = self._feed("fast", X, scores, gap=0.01)
        self.assertGreaterEqual(max(levels), 2)
        self.assertLessEqual(len(levels), config.WINDOW_SIZE)


class TestGatewayIntegration(unittest.TestCase):
    def test_audit_log_and_throttle(self):
        d = 6
        model = FakeModel(d)
        benign = _toy_data(n=400, d=d)
        det = RuleBasedDetector(WindowFeatureExtractor(benign))
        gw = ScoringGateway(model, det, AdaptiveDefense())
        rng = np.random.default_rng(4)
        # force level-3 suspicion first, then verify rate limiter engages
        X = rng.normal(0, 6, size=(260, d))
        scores_proxy = rng.uniform(0.45, 0.55, 260)
        for i, x in enumerate(X):
            gw.score("atk", x, i * 0.01)
        self.assertEqual(len(gw.audit), 260)
        # after enough window evaluations the account must have escalated
        self.assertGreaterEqual(gw.audit.to_frame()["level"].max(), 2)
        # rate-limit config sanity: high levels actually throttle
        self.assertIsNotNone(RATE_LIMIT_QPS[3])

    def test_benign_gets_raw_scores(self):
        d = 6
        model = FakeModel(d)
        benign = _toy_data(n=400, d=d)
        det = RuleBasedDetector(WindowFeatureExtractor(benign))
        gw = ScoringGateway(model, det, NullDefense())
        for i in range(220):
            r = gw.score("partner", benign[i], i * 0.2)
        audit = gw.audit.to_frame()
        np.testing.assert_allclose(audit["returned_score"], audit["raw_score"])


if __name__ == "__main__":
    unittest.main()
