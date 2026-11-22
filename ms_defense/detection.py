"""Streaming detectors: rule-based weighted scorer and supervised ML variant,
plus cross-account correlation for coordinated / distributed extraction."""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from . import config
from .detection_features import WindowFeatureExtractor


class _AccountState:
    __slots__ = ("X", "scores", "labels", "ts", "suspicion_ema", "level",
                 "raw_window_scores", "coarse_cells", "behavior_profile")

    def __init__(self):
        self.X: deque = deque(maxlen=config.WINDOW_SIZE * 3)
        self.scores: deque = deque(maxlen=config.WINDOW_SIZE * 3)
        self.labels: deque = deque(maxlen=config.WINDOW_SIZE * 3)
        self.ts: deque = deque(maxlen=config.WINDOW_SIZE * 3)
        self.suspicion_ema = 0.0
        self.level = 0
        self.raw_window_scores: list[float] = []
        self.coarse_cells: set[int] = set()
        # EMA of the strongest behavioral extraction signals
        self.behavior_profile = {"boundary_fraction": 0.0, "novelty": 0.0}


class RuleBasedDetector:
    """Weighted, range-normalized behavioral scorer over sliding windows.

    Weights encode the relative selectivity of each signal: boundary hunting
    and bisection-style coordinate walks are nearly unique to extraction,
    while raw volume is deliberately given little weight (benign partners are
    high-volume too).
    """

    WEIGHTS = {
        "boundary_fraction": 2.0,
        "tail_fraction": 1.4,
        "novelty": 2.2,
        "pair_diversity": 0.8,
        "dedup_ratio": 0.4,
        "label_flip_rate": 0.8,
        "score_entropy": 0.6,
        "score_range": 0.4,
        "rate_burst": 1.8,
        "sign_flip_rate": 1.6,
        "coord_concentration": 1.4,
        "span_coverage": 0.8,
    }

    # benign-calibrated upper anchors -> score 1.0 at/above anchor.
    # Fixed values sit ~2x above the benign 99th percentile observed on
    # replay windows, so benign windows land around 0.35-0.45 and extraction
    # windows (which saturate the manifold signals) reach 0.75+. The two
    # manifold anchors are fit at runtime from the benign pool.
    ANCHORS = {
        "boundary_fraction": 0.60,
        "tail_fraction": 0.35,
        "novelty": None,          # set at fit time from benign data
        "pair_diversity": None,
        "dedup_ratio": 1.0,
        "label_flip_rate": 0.70,
        "score_entropy": 1.05,
        "score_range": 1.05,
        "rate_burst": 0.80,
        "sign_flip_rate": 0.75,
        "coord_concentration": 0.40,
        "span_coverage": 2.0,
    }

    def __init__(self, extractor: WindowFeatureExtractor):
        self.extractor = extractor
        self.accounts: dict[str, _AccountState] = defaultdict(_AccountState)
        self._last_eval_at: dict[str, int] = {}
        self._fit_anchors(extractor.benign_pool)
        self.total_weight = sum(self.WEIGHTS.values())
        # for cross-account correlation
        self.signature_sets: dict[str, set[int]] = defaultdict(set)
        self.clusters: list[list[str]] = []
        self._n_obs = 0
        self._last_pair_scores: list[tuple] = []

    def _fit_anchors(self, benign_pool: np.ndarray) -> None:
        # calibrate manifold anchors at 1.5x the benign 99th percentile, so a
        # normal window scores ~0.65 on these signals and OOD probing saturates
        rng = np.random.default_rng(0)
        nov, div = [], []
        for _ in range(40):
            idx = rng.integers(0, len(benign_pool), config.WINDOW_SIZE)
            Xw = benign_pool[idx]
            nov.append(self.extractor.novelty(Xw))
            div.append(self.extractor.pair_diversity(Xw))
        self.ANCHORS["novelty"] = max(np.percentile(nov, 99) * 2.0, 1e-6)
        self.ANCHORS["pair_diversity"] = max(np.percentile(div, 99) * 1.4, 1e-6)

    def observe(self, account_id: str, ts: float, x: np.ndarray, score: float):
        st = self.accounts[account_id]
        st.X.append(x)
        st.scores.append(score)
        st.labels.append(int(score >= config.RISK_THRESHOLD))
        st.ts.append(ts)
        n = len(st.X)
        last = self._last_eval_at.get(account_id, 0)
        is_early = (n == config.EARLY_WINDOW and last == 0)
        is_regular = (n >= config.WINDOW_SIZE and (n - last) >= config.WINDOW_STRIDE)
        if is_early or is_regular:
            self._last_eval_at[account_id] = n
            wsize = min(n, config.WINDOW_SIZE)
            X = np.array(st.X)[-wsize:]
            s = np.array(st.scores)[-wsize:]
            lab = np.array(st.labels)[-wsize:]
            t = np.array(st.ts)[-wsize:]
            feats = self.extractor.extract(X, s, lab, t)
            raw = self._score_window(feats)
            if last == 0 or st.suspicion_ema == 0.0:
                # first window (early or full-size): initialize from evidence
                # instead of pretending the account has an unseen clean history
                st.suspicion_ema = raw
            else:
                st.suspicion_ema = (
                    config.SUSPICION_EMA_ALPHA * raw
                    + (1 - config.SUSPICION_EMA_ALPHA) * st.suspicion_ema
                )
            st.level = self._to_level(st.suspicion_ema)
            st.raw_window_scores.append(raw)
            for name in ("boundary_fraction", "novelty"):
                idx = self.extractor.FEATURE_NAMES.index(name)
                prev = st.behavior_profile[name]
                st.behavior_profile[name] = 0.5 * feats[idx] + 0.5 * prev
            for h in self.extractor.hashed_bin_signatures(X):
                self.signature_sets[account_id].add(int(h))
            st.coarse_cells |= self.extractor.coarse_cell_set(X)

        # periodic cross-account coordination analysis + joint escalation
        self._n_obs += 1
        if self._n_obs % 500 == 0:
            self._correlate_and_escalate()
        return st.suspicion_ema, st.level

    def _is_behaviorally_extractive(self, acct: str) -> bool:
        """An account only qualifies for cluster membership when its own
        window behavior shows extraction signals — spatial overlap alone is
        meaningless because all partners operate on the same business
        manifold. Thresholds sit just above benign 99th percentiles."""
        prof = self.accounts[acct].behavior_profile
        return (prof["boundary_fraction"] >= 0.34
                or prof["novelty"] >= self.ANCHORS["novelty"] * 0.8
                or self.accounts[acct].suspicion_ema >= 0.55)

    def _cadence(self, acct: str) -> float:
        """Median inter-query gap. Coordinated machine sweeps hold a near
        constant sub-100ms cadence; benign Poisson traffic is much slower and
        over-dispersed."""
        ts = np.array(self.accounts[acct].ts)
        if len(ts) < 5:
            return 1.0
        return float(np.median(np.diff(ts)))

    def _marginal_corr(self, a: str, b: str) -> float:
        """Median across-feature correlation of per-account query histograms.
        Accounts driven by the same sampling machinery share marginal
        distributions even when no exact cell coincides (high-d sparsity)."""
        Xa = np.array(self.accounts[a].X)
        Xb = np.array(self.accounts[b].X)
        if len(Xa) < 50 or len(Xb) < 50:
            return 0.0
        corrs = []
        lo = self.extractor.benign_min
        rng_ = self.extractor.benign_max - lo + 1e-9
        for j in range(Xa.shape[1]):
            ha, _ = np.histogram((Xa[:, j] - lo[j]) / rng_[j],
                                 bins=8, range=(0, 1), density=True)
            hb, _ = np.histogram((Xb[:, j] - lo[j]) / rng_[j],
                                 bins=8, range=(0, 1), density=True)
            if ha.std() > 1e-9 and hb.std() > 1e-9:
                corrs.append(float(np.corrcoef(ha, hb)[0, 1]))
        return float(np.median(corrs)) if corrs else 0.0

    def _coordination_score(self, a: str, b: str) -> float:
        """Composite evidence that two already-suspicious accounts are operated
        together. Spatial cell overlap is deliberately only a weak component:
        in 24-d space even related attackers rarely hit the same coarse cell,
        while benign partners overlap everyone on the business manifold."""
        pa, pb = self.accounts[a].behavior_profile, self.accounts[b].behavior_profile
        nov_anchor = self.ANCHORS["novelty"]
        # both machine-paced -> strong shared signal
        cadence_match = 1.0 if (self._cadence(a) < 0.12 and self._cadence(b) < 0.12) else 0.0
        profile_sim = float(np.exp(
            -abs(pa["boundary_fraction"] - pb["boundary_fraction"])
            -abs(pa["novelty"] - pb["novelty"]) / (nov_anchor + 1e-9)))
        marg = max(0.0, self._marginal_corr(a, b))
        ca, cb = self.accounts[a].coarse_cells, self.accounts[b].coarse_cells
        cell_overlap = len(ca & cb) / min(len(ca), len(cb)) if ca and cb else 0.0
        return 0.35 * cadence_match + 0.25 * profile_sim + 0.25 * marg + 0.15 * cell_overlap

    def _correlate_and_escalate(self) -> None:
        """Union-find behaviorally-extractive accounts whose composite
        coordination score is high, then jointly escalate the cluster. Each
        individual account was kept below the alert threshold by splitting the
        query budget across the cluster — pooling evidence defeats that."""
        eligible = [a for a, st in self.accounts.items()
                    if len(st.coarse_cells) >= 20 and self._is_behaviorally_extractive(a)]
        parent = {a: a for a in eligible}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            parent[find(a)] = find(b)

        pair_scores = []
        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                score = self._coordination_score(eligible[i], eligible[j])
                pair_scores.append((eligible[i], eligible[j], score))
                if score >= 0.55:
                    union(eligible[i], eligible[j])
        self._last_pair_scores = pair_scores

        groups: dict[str, list[str]] = defaultdict(list)
        for a in eligible:
            groups[find(a)].append(a)
        self.clusters = [g for g in groups.values() if len(g) >= 2]
        for group in self.clusters:
            pool = [s for a in group for s in self.accounts[a].raw_window_scores[-3:]]
            if not pool:
                continue
            cluster_score = float(np.mean(pool))
            n_mild = sum(1 for a in group
                         if self.accounts[a].suspicion_ema >= 0.35)
            if n_mild >= 2 and cluster_score >= 0.55:
                boost = min(0.30, 0.10 + 0.05 * (len(group) - 2))
                for a in group:
                    st = self.accounts[a]
                    st.suspicion_ema = min(0.95, st.suspicion_ema + boost)
                    st.level = self._to_level(st.suspicion_ema)

    def _score_window(self, feats: np.ndarray) -> float:
        total = 0.0
        for name, val in zip(self.extractor.FEATURE_NAMES, feats):
            anchor = self.ANCHORS[name]
            frac = np.clip(val / anchor, 0, 1.5) if anchor else 0.0
            total += self.WEIGHTS[name] * min(frac, 1.0)
        return float(total / self.total_weight)

    @staticmethod
    def _to_level(suspicion: float) -> int:
        if suspicion >= 0.80:
            return 3
        if suspicion >= config.DETECT_ALERT:
            return 2
        if suspicion >= 0.55:
            return 1
        return 0

    # -- cross-account coordination analysis --------------------------------
    def correlated_accounts(self, min_score: float = 0.55) -> list[tuple[str, str, float]]:
        # only behaviorally-extractive accounts are eligible for correlation;
        # linkage uses the composite coordination score rather than raw cell
        # overlap, which is unreliable in the 24-d sparse feature space
        eligible = [a for a in self.accounts if self._is_behaviorally_extractive(a)]
        out = []
        for i in range(len(eligible)):
            for j in range(i + 1, len(eligible)):
                score = self._coordination_score(eligible[i], eligible[j])
                if score >= min_score:
                    out.append((eligible[i], eligible[j], round(score, 4)))
        return sorted(out, key=lambda t: -t[2])


class MLDetector(RuleBasedDetector):
    """Same windowing, but the window score comes from a supervised model
    trained on benign-vs-attack window feature vectors."""

    def __init__(self, extractor: WindowFeatureExtractor):
        super().__init__(extractor)
        self.clf = RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=config.RANDOM_STATE,
            class_weight="balanced",
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLDetector":
        """X: window feature vectors, y: 0=benign / 1=extraction windows."""
        self.clf.fit(X, y)
        self._fitted = True
        return self

    def _score_window(self, feats: np.ndarray) -> float:
        if not self._fitted:
            return super()._score_window(feats)
        return float(self.clf.predict_proba(feats.reshape(1, -1))[0, 1])


class NoveltyDetector:
    """Unsupervised fallback: IsolationForest directly on query vectors,
    fused with the behavioral score (catches OOD probing without labels)."""

    def __init__(self, benign_pool: np.ndarray, contamination: float = 0.08):
        self.iforest = IsolationForest(
            n_estimators=200, contamination=contamination,
            random_state=config.RANDOM_STATE,
        ).fit(benign_pool)

    def ood_fraction(self, X: np.ndarray) -> float:
        preds = self.iforest.predict(X)
        return float(np.mean(preds == -1))
