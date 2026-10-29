"""How do the conclusions depend on how detectable the simulated incipient fault is?

Re-runs three models for several initial defect amplitudes a0 (in units of noise sigma).
    python scripts/sensitivity_a0.py [--out results]
"""
import argparse, os, sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from imsae.pipeline import MODELS, fit_and_score, sim_split, test_at_operating_points
from imsae.simulate import SimConfig

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); a = ap.parse_args()
rows = []
for a0 in (0.25, 0.5, 1.0):
    data = sim_split(cfg=SimConfig(a0=a0))               # no cache: features depend on a0
    for name in ("dense_wide", "dense_narrow", "lstm_dae"):
        for seed in (0, 1):
            sc, _ = fit_and_score(MODELS[name], seed, data)
            r, _ = test_at_operating_points(sc)
            c = r["cost"]["test"]
            rows.append(dict(a0=a0, model=name, seed=seed, auc=r["auc"], recall=c["recall"], fa_per_day=c["fa_per_day"],
                             in_time=f'{c["detected"]}/{c["events"]}', mean_delay=c["mean_delay"], cost=c["cost"]))
            print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv(Path(a.out) / "sensitivity_a0_per_seed.csv", index=False)
df.groupby(["a0", "model"]).mean(numeric_only=True).drop(columns="seed").to_csv(Path(a.out) / "sensitivity_a0.csv")
print("wrote sensitivity")
