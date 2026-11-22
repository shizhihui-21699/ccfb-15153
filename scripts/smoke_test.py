"""Quick smoke test: small budgets, one of each component."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ms_defense.data import load_dataset
from ms_defense.victim import load_model
from ms_defense.experiment import build_context, make_gateway, run_single_account_attacks, evaluate_stolen_set
from ms_defense.detection import RuleBasedDetector
from ms_defense.defense import AdaptiveDefense, NullDefense

ds = load_dataset()
victim, _, _ = load_model()
ctx = build_context(ds, victim)

for defense_cls, tag in [(NullDefense, "null"), (AdaptiveDefense, "adaptive")]:
    det = RuleBasedDetector(ctx.extractor)
    gw = make_gateway(ctx, det, defense_cls())
    attacks = run_single_account_attacks(ctx, gw, budget=300, t0=1_000.0)
    print(f"\n--- {tag} ---")
    for name, fr in attacks.items():
        print(name, len(fr), "rows;",
              "attack rows:", (fr["origin"] == "attack").sum())
        fid = evaluate_stolen_set(ctx, fr)
        if fid:
            print("   agreement=%.3f rmse=%.3f surrogate_auc=%.3f bypass=%.3f"
                  % (fid["label_agreement"], fid["score_rmse"],
                     fid["surrogate_auc"], fid["evasion_bypass_rate_vs_victim"]))
    audit = gw.audit.to_frame()
    print("levels seen:", audit["level"].value_counts().to_dict())
    print("mean suspicion by account:")
    print(audit.groupby("account_id")["suspicion"].mean().round(3).to_dict())
print("\nsmoke OK")
