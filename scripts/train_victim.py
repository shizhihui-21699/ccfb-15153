"""Train and persist the victim risk-control model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ms_defense import config
from ms_defense.data import load_dataset
from ms_defense.victim import train_victim, evaluate, save_model


def main() -> None:
    ds = load_dataset()
    print(f"[victim] dataset={ds.name} "
          f"train={len(ds.y_train)} val={len(ds.y_val)} test={len(ds.y_test)} "
          f"positive_rate={ds.y_train.mean():.3f}")
    model = train_victim(ds)
    metrics = evaluate(model, ds)
    path = save_model(model, ds)
    (config.REPORT_DIR / "victim_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[victim] saved -> {path}")
    for k, v in metrics.items():
        print(f"         {k:10s} = {v:.4f}")


if __name__ == "__main__":
    main()
