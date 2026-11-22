"""Diagnose window-feature separation between benign and attack windows."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ms_defense import config
from ms_defense.data import load_dataset
from ms_defense.victim import load_model
from ms_defense.experiment import build_context, make_gateway, run_benign_traffic, run_single_account_attacks
from ms_defense.detection import RuleBasedDetector
from ms_defense.defense import NullDefense

ds = load_dataset()
victim, _, _ = load_model()
ctx = build_context(ds, victim)
ext = ctx.extractor

det = RuleBasedDetector(ext)
gw = make_gateway(ctx, det, NullDefense())

benign = run_benign_traffic(ctx, gw, n_partners=3, queries_each=800)
attacks = run_single_account_attacks(ctx, gw, budget=1200, t0=2_000_000.0)

feat_cols = [c for c in benign.columns if c.startswith("f")]


def windows(frame):
    out = []
    for acct, g in frame.groupby("account_id"):
        g = g.sort_values("ts")
        X = g[feat_cols].to_numpy(); s = g["score"].to_numpy()
        l = g["label"].to_numpy(); t = g["ts"].to_numpy()
        for start in range(0, len(X) - config.WINDOW_SIZE + 1, config.WINDOW_STRIDE):
            sl = slice(start, start + config.WINDOW_SIZE)
            out.append(ext.extract(X[sl], s[sl], l[sl], t[sl]))
    return np.vstack(out)

ben_w = windows(benign)
rows = []
rows.append(pd.Series(ben_w.mean(axis=0), index=ext.FEATURE_NAMES).rename("benign_mean"))
rows.append(pd.Series(np.percentile(ben_w, 99, axis=0), index=ext.FEATURE_NAMES).rename("benign_p99"))
for name, fr in attacks.items():
    w = windows(fr)
    rows.append(pd.Series(w.mean(axis=0), index=ext.FEATURE_NAMES).rename(name))
df = pd.DataFrame(rows).T
print(df.to_string(float_format=lambda v: f"{v:7.3f}"))
print("\nanchors:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in det.ANCHORS.items()})
