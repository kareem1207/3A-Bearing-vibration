"""Run every experiment for the report and write tables/figures to results/.

    python scripts/run_experiments.py                    # simulated IMS-style data
    python scripts/run_experiments.py --quick            # tiny smoke run
    python scripts/run_experiments.py --data ims --ims-root /path/to/IMS   # real data
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imsae import evaluate as ev
from imsae.pipeline import (MODELS, SEEDS, _pairs, fit_and_score, ims_split, sim_split,
                            test_at_operating_points)
from imsae.simulate import SimConfig

ap = argparse.ArgumentParser()
ap.add_argument("--data", choices=["sim", "ims"], default="sim")
ap.add_argument("--ims-root")
ap.add_argument("--out", default="results")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
EPOCHS = 3 if args.quick else 100
seeds = SEEDS[:1] if args.quick else SEEDS

if args.data == "sim":
    cfg = SimConfig(n_files=120) if args.quick else SimConfig()
    data = sim_split(2 if args.quick else 3, 1 if args.quick else 2, 1 if args.quick else 3,
                     cfg=cfg, cache=OUT / "cache")
else:
    data = ims_split(args.ims_root, limit=150 if args.quick else None)
D = data["train"][0].X.shape[1]
LABEL = "SIMULATED" if args.data == "sim" else "REAL IMS"
print(f"[{LABEL}] train/val/test series:", {k: len(v) for k, v in data.items()}, "features:", D)


def flat(res, prefix):
    r = res["test"]
    return {f"{prefix}_{k}": r[k] for k in ("recall", "precision", "f1", "fa_per_day", "detected", "events", "mean_delay", "cost")}


# ---------------- E2: model comparison ----------------
rows, store = [], {}
for name, spec in MODELS.items():
    for seed in seeds:
        print("fit", name, seed, flush=True)
        scores, hist = fit_and_score(spec, seed, data, EPOCHS)
        res, ops = test_at_operating_points(scores)
        store[(name, seed)] = (scores, hist, res, ops)
        rows.append(dict(model=name, seed=seed, m=res["m"], auc=res["auc"], epochs=len(hist["loss"]),
                         val_cost=res["cost"]["val"]["cost"],
                         **flat(res["cost"], "cost_pt"), **flat(res["f1"], "f1_pt"),
                         thr_cost=res["cost"]["val"]["thr"], thr_f1=res["f1"]["val"]["thr"]))
df = pd.DataFrame(rows)
df.to_csv(OUT / "model_comparison_per_seed.csv", index=False)
num = df.drop(columns=["seed"]).groupby("model").agg(["mean", "std"])
num.columns = ["_".join(c) for c in num.columns]
num.to_csv(OUT / "model_comparison_summary.csv")
print(df.groupby("model")[["auc", "cost_pt_recall", "cost_pt_fa_per_day", "cost_pt_cost", "f1_pt_f1"]].mean().round(3))

main = df.groupby("model")["val_cost"].mean().idxmin()
print("main model (lowest mean validation cost):", main)
scores, hist, res, ops = store[(main, seeds[0])]

# ---------------- E1: bottleneck sweep (dense AE) ----------------
b_rows = []
for b in ([D, 8, 2] if args.quick else [D, 32, 16, 8, 4, 2]):
    for seed in seeds:
        spec = dict(kind="dense", bottleneck="input" if b == D else b, hidden=64)
        sc, h = fit_and_score(spec, seed, data, EPOCHS)
        test = _pairs(sc["test"])
        err_n = np.mean(np.concatenate([s[:o if o is not None else len(s)] for s, o in test]))
        err_f = np.mean(np.concatenate([s[o:] for s, o in test if o is not None]))
        b_rows.append(dict(bottleneck=b, seed=seed, auc=ev.pooled_auc(test), err_healthy=err_n,
                           err_fault=err_f, ratio=err_f / err_n, train_loss=h["loss"][-1], val_loss=h["val_loss"][-1]))
bdf = pd.DataFrame(b_rows); bdf.to_csv(OUT / "bottleneck_sweep_per_seed.csv", index=False)
bsum = bdf.groupby("bottleneck").mean(numeric_only=True).drop(columns="seed"); bsum.to_csv(OUT / "bottleneck_sweep.csv")
print(bsum.round(4))

# ---------------- E3: sequence length ablation (LSTM AE) ----------------
t_rows = []
for T in ([1, 6] if args.quick else [1, 3, 6, 12]):
    for seed in seeds:
        spec = dict(MODELS["lstm_dae"], T=T)
        sc, _ = fit_and_score(spec, seed, data, EPOCHS)
        r, _ = test_at_operating_points(sc)
        t_rows.append(dict(T=T, seed=seed, auc=r["auc"], m=r["m"], **flat(r["cost"], "cost_pt")))
tdf = pd.DataFrame(t_rows); tdf.to_csv(OUT / "sequence_length_per_seed.csv", index=False)
tsum = tdf.groupby("T").mean(numeric_only=True).drop(columns="seed"); tsum.to_csv(OUT / "sequence_length.csv")
print(tsum.round(3))

# ---------------- cost table for the main model ----------------
rows_cost = []
for label, key in (("Best F1 (validation)", "f1"), ("Minimum cost (validation)", "cost")):
    t = res[key]["test"]
    rows_cost.append({"operating point": label, "threshold": t["thr"], "alarm rule (m)": res["m"],
                      "recall": t["recall"], "precision": t["precision"], "F1": t["f1"],
                      "false alarms/day": t["fa_per_day"], "faults detected in time": f"{t['detected']}/{t['events']}",
                      "missed faults": t["missed"], "false alarm episodes": t["fa"],
                      "downtime cost Rs": ev.COST_MISS * t["missed"], "inspection cost Rs": ev.COST_FALSE_ALARM * t["fa"],
                      "total cost Rs": t["cost"]})
cdf = pd.DataFrame(rows_cost); cdf.to_csv(OUT / "cost_analysis.csv", index=False)
print(cdf.T)

# ---------------- figures ----------------
test_pairs = _pairs(scores["test"])
rows_m = ev.sweep(test_pairs, res["m"])
P = np.array([r["precision"] for r in rows_m]); R = np.array([r["recall"] for r in rows_m])
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(R, P, color="tab:blue", lw=1.8, label=f"{main} (test, alarm rule m={res['m']})")
for key, mk, col, lab in (("f1", "o", "tab:green", "best-F1 threshold"), ("cost", "*", "tab:red", "min-cost threshold")):
    t = res[key]["test"]
    ax.scatter([t["recall"]], [t["precision"]], marker=mk, s=140 if mk == "*" else 70, color=col, zorder=5, label=lab)
ax.axvline(0.85, ls="--", color="gray", lw=1); ax.text(0.855, 0.05, "recall target 0.85", color="gray", fontsize=8)
ax.set_xlabel("Recall (fault readings flagged)"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02); ax.set_xlim(0, 1.02)
ax.set_title(f"Precision-recall curve [{LABEL} data]"); ax.legend(loc="lower left", fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "fig_pr_curve.png", dpi=160); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, sp in zip(axes, ("val", "test")):
    rws = ev.sweep(_pairs(scores[sp]), res["m"]); thr = [r["thr"] for r in rws]
    ax.semilogy(thr, [max(r["cost"], 1) for r in rws], color="k")
    for key, col in (("f1", "tab:green"), ("cost", "tab:red")):
        ax.axvline(res[key]["val"]["thr"], color=col, ls="--", label=f"{key} threshold")
    ax.set_title(f"{sp}: total cost vs threshold"); ax.set_xlabel("threshold (reconstruction error)"); ax.grid(alpha=.3)
axes[0].set_ylabel("cost (Rs, log)"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_cost_vs_threshold.png", dpi=160); plt.close(fig)

fi = next(i for i, (s, o, n) in enumerate(scores["test"]) if o is not None)
s, o, n = scores["test"][fi]
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.semilogy(s, lw=.8, label="reconstruction error"); ax.axvline(o, color="k", ls=":", label="fault onset (label)")
for key, col in (("f1", "tab:green"), ("cost", "tab:red")):
    ax.axhline(res[key]["val"]["thr"], color=col, ls="--", label=f"{key} threshold")
ax.set_title(f"Score timeline, {n} ({main}) [{LABEL}]"); ax.set_xlabel("reading (10 min apart)"); ax.legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(OUT / "fig_timeline.png", dpi=160); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
g = bdf.groupby("bottleneck")
axes[0].errorbar(g["auc"].mean().index, g["auc"].mean(), g["auc"].std(), marker="o")
axes[0].set_xscale("log", base=2); axes[0].set_xlabel("bottleneck width (input width = %d)" % D); axes[0].set_ylabel("ROC-AUC healthy vs fault"); axes[0].grid(alpha=.3)
axes[1].errorbar(g["ratio"].mean().index, g["ratio"].mean(), g["ratio"].std(), marker="o", color="tab:red")
axes[1].set_xscale("log", base=2); axes[1].set_yscale("log"); axes[1].set_xlabel("bottleneck width"); axes[1].set_ylabel("fault / healthy mean error"); axes[1].grid(alpha=.3)
fig.suptitle(f"Dense AE bottleneck sweep [{LABEL}]"); fig.tight_layout(); fig.savefig(OUT / "fig_bottleneck.png", dpi=160); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, nm in zip(axes, ("dense_wide", "dense_narrow", main if main.startswith("lstm") else "lstm_dae")):
    h = store[(nm, seeds[0])][1]
    ax.plot(h["loss"], label="train"); ax.plot(h["val_loss"], label="validation (healthy)"); ax.set_title(nm); ax.set_xlabel("epoch"); ax.set_yscale("log"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_training_curves.png", dpi=160); plt.close(fig)

tg = tdf.groupby("T")
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.errorbar(tg["auc"].mean().index, tg["auc"].mean(), tg["auc"].std(), marker="o"); ax.set_xlabel("window length T (readings)"); ax.set_ylabel("ROC-AUC")
ax.set_title(f"LSTM AE window length [{LABEL}]"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(OUT / "fig_sequence_length.png", dpi=160); plt.close(fig)

summary = dict(data=LABEL, main_model=main, features=D, series={k: len(v) for k, v in data.items()},
               targets=ev.TARGETS, operating_m=res["m"], auc=res["auc"],
               best_f1_point=res["f1"]["test"], min_cost_point=res["cost"]["test"])
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
print("wrote", OUT)
