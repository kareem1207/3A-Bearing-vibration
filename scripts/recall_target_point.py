"""Cost of meeting the recall target: the threshold chosen on VALIDATION data that reaches
recall >= 0.85 with the fewest false alarms, applied unchanged to the test runs.
Refits the models (3 seeds), so numbers can differ slightly from run_experiments.py."""
import argparse, os, sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from imsae import evaluate as ev
from imsae.pipeline import MODELS, SEEDS, _pairs, fit_and_score, sim_split

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); a = ap.parse_args()
data = sim_split(cache=Path(a.out) / "cache")
rows = []
for name in ("dense_narrow", "dense_dae_sparse", "lstm_dae"):
    for seed in SEEDS:
        sc, _ = fit_and_score(MODELS[name], seed, data)
        val, test = _pairs(sc["val"]), _pairs(sc["test"])
        for m in (1, 2):
            ok = [r for r in ev.sweep(val, m) if r["recall"] >= 0.85]
            if not ok:
                rows.append(dict(model=name, seed=seed, m=m, reached=False)); continue
            thr = max(r["thr"] for r in ok)                      # highest threshold still reaching the target
            t = ev.evaluate(test, thr, m)
            rows.append(dict(model=name, seed=seed, m=m, reached=True, thr=thr, recall=t["recall"], precision=t["precision"],
                             fa_per_day=t["fa_per_day"], in_time=t["detected"], events=t["events"],
                             mean_delay=t["mean_delay"], cost=t["cost"]))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv(Path(a.out) / "recall_target_point_per_seed.csv", index=False)
df.groupby(["model", "m"]).mean(numeric_only=True).drop(columns="seed").to_csv(Path(a.out) / "recall_target_point.csv")
print("wrote recall target")
