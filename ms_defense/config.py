"""Global configuration and paths for the model-stealing defense lab."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_store"
MODEL_DIR = ROOT / "artifacts" / "models"
REPORT_DIR = ROOT / "artifacts" / "reports"
FIG_DIR = REPORT_DIR / "figures"
for d in (DATA_DIR, MODEL_DIR, REPORT_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Victim / data -------------------------------------------------------
DATASET = os.environ.get("MS_DATASET", "german")  # german | synthetic
RANDOM_STATE = 42
TEST_SIZE = 0.25
VAL_SIZE = 0.15
RISK_THRESHOLD = 0.5  # decision threshold used by the published API

# German Credit (Statlog, UCI) — public benchmark, 24 numeric features
GERMAN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data-numeric"
)

# ---- Detection windows ---------------------------------------------------
WINDOW_SIZE = 200          # queries per sliding window
WINDOW_STRIDE = 50         # recompute features every N new queries
EARLY_WINDOW = 80          # first evaluation sooner: closes the cold-start hole
BOUNDARY_BAND = 0.10       # |score-0.5| <= band counts as boundary query
TAIL_QUANTILE = 0.05       # extreme-tail threshold for coverage features
SUSPICION_EMA_ALPHA = 0.5  # EMA smoothing of window suspicion
DETECT_ALERT = 0.62        # suspicion level above which defenses escalate

# ---- Defense policy ------------------------------------------------------
SALT_ROTATION_QUERIES = 50_000  # rotate deterministic-noise salt periodically
QUANT_GRID = [0.0, 0.01, 0.05, 0.10]   # per-level score quantization step
NOISE_SD = [0.0, 0.002, 0.02, 0.05]    # per-level deterministic noise scale
RATE_LIMIT_QPS = [None, None, 5.0, 1.0]  # per-level soft throttle
# deterministic, boundary-targeted label poisoning probabilities per level.
# only flips scores already close to the threshold, so the attacker's hard
# labels (which train the surrogate) are corrupted exactly where they matter.
LABEL_POISON_P = [0.0, 0.0, 0.15, 0.30]
POISON_BAND = 0.15                      # only queries within |score-0.5| of boundary
POISON_MARGIN = [0.0, 0.0, 0.08, 0.15]  # how far past the threshold a flip lands

# ---- Attack budgets ------------------------------------------------------
ATTACK_BUDGETS = [200, 500, 1_000, 2_000, 4_000, 8_000]
N_BENIGN_PARTNERS = 6
BENIGN_QUERIES_PER_PARTNER = 1_500
N_ATTACK_ACCOUNTS = 5
