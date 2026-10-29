"""Build notebooks/DL_T03_3A.ipynb (kept as code so the notebook stays in sync with the package)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
md("""# DL_T03_3A: Autoencoder anomaly detection on bearing vibration (NASA IMS)
Team 3, variant 3A. Task 3 of the CM701PC / CM751PC Complex Engineering Problem.

> **Data status.** `USE_REAL_IMS = False` runs on a *simulated* IMS style run-to-failure dataset
> (same file shape, 20 kHz, 10 minute spacing). Numbers from it describe the pipeline, **not** real bearings.
> To reproduce on real data download the NASA IMS archive, set `USE_REAL_IMS = True` and `IMS_ROOT`.
> The onset labels for real data are an assumption (see `src/imsae/ims_io.py`)."""),
md("## 0. Setup"),
code("""import os, sys, subprocess
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if os.path.exists("../src/imsae"):             # started from notebooks/
    os.chdir("..")
if not os.path.exists("src/imsae"):            # running in Colab: fetch the repository first
    subprocess.run(["git", "clone", "https://github.com/kareem1207/3a-bearing-vibration.git"], check=True)
    os.chdir("3a-bearing-vibration")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tensorflow-cpu", "scikit-learn", "pandas", "matplotlib"], check=True)
sys.path.insert(0, "src")

USE_REAL_IMS = False
IMS_ROOT = "/content/IMS"      # folder that holds 1st_test, 2nd_test, 3rd_test
QUICK = False                  # True: 3 epochs, one seed (smoke run). False: full training (about 6 minutes)"""),
md("## 1. Data and features\nEvery snapshot (1 s, 20480 samples per channel) becomes 11 features per bearing: log RMS, kurtosis, crest factor and 8 band powers. Each series is standardised with its **own first 40 percent** (assumed healthy); no fault labels are used for training."),
code("""import numpy as np, matplotlib.pyplot as plt, pandas as pd
from imsae import evaluate as ev
from imsae.features import FEATURE_NAMES
from imsae.pipeline import MODELS, SEEDS, _pairs, fit_and_score, ims_split, sim_split, test_at_operating_points
from imsae.simulate import SimConfig

if USE_REAL_IMS:
    data = ims_split(IMS_ROOT, limit=150 if QUICK else None)
else:
    cfg = SimConfig(n_files=120) if QUICK else SimConfig()
    data = sim_split(*( (2,1,1) if QUICK else (3,2,3) ), cfg=cfg, cache="results/cache")
print({k: len(v) for k, v in data.items()}, "series; features:", len(FEATURE_NAMES))
s = next(x for x in data["test"] if x.onset is not None)
fig, ax = plt.subplots(1, 3, figsize=(12, 3))
for a, j in zip(ax, (0, 1, 7)):
    a.plot(s.X[:, j]); a.axvline(s.onset, color="k", ls=":"); a.set_title(FEATURE_NAMES[j] + " (standardised)")
plt.tight_layout(); plt.show()"""),
md("## 2. Why a bottleneck as wide as the input fails (report Q1)\nSame dense autoencoder, bottleneck = input width versus a narrow one."),
code("""EPOCHS = 3 if QUICK else 100
out = {}
for name in ("dense_wide", "dense_narrow"):
    sc, h = fit_and_score(MODELS[name], 0, data, EPOCHS)
    t = _pairs(sc["test"])
    err_h = np.mean(np.concatenate([x[:o if o is not None else len(x)] for x, o in t]))
    err_f = np.mean(np.concatenate([x[o:] for x, o in t if o is not None]))
    out[name] = dict(auc=ev.pooled_auc(t), healthy_err=err_h, fault_err=err_f, ratio=err_f / err_h)
pd.DataFrame(out).T.round(3)"""),
md("## 3. Four models, thresholds chosen on validation data only"),
code("""rows, store = [], {}
for name, spec in MODELS.items():
    sc, h = fit_and_score(spec, 0, data, EPOCHS)
    res, ops = test_at_operating_points(sc)
    store[name] = (sc, res)
    c = res["cost"]["test"]; f = res["f1"]["test"]
    rows.append(dict(model=name, auc=res["auc"], recall=c["recall"], fa_per_day=c["fa_per_day"],
                     detected=f'{c["detected"]}/{c["events"]}', cost_Rs=c["cost"], best_F1=f["f1"]))
pd.DataFrame(rows).round(3)"""),
md("## 4. Precision-recall curve with both operating points, and the cost table"),
code("""name = "lstm_dae"
sc, res = store[name]
rows_m = ev.sweep(_pairs(sc["test"]), res["m"])
P = [r["precision"] for r in rows_m]; R = [r["recall"] for r in rows_m]
plt.figure(figsize=(6, 4.5)); plt.plot(R, P, label=name)
for key, mk, col in (("f1", "o", "tab:green"), ("cost", "*", "tab:red")):
    t = res[key]["test"]; plt.scatter([t["recall"]], [t["precision"]], marker=mk, s=120, color=col, label=key + " threshold", zorder=5)
plt.xlabel("recall"); plt.ylabel("precision"); plt.legend(); plt.grid(alpha=.3); plt.show()
pd.DataFrame({k: {kk: res[k]["test"][kk] for kk in ("thr","recall","precision","f1","fa_per_day","missed","fa","cost")} for k in ("f1","cost")}).T.round(3)"""),
md("The full multi seed experiments, the bottleneck sweep, the window length ablation and all figures are produced by `python scripts/run_experiments.py` (outputs in `results/`)."),
]
nbf.write(nb, "notebooks/DL_T03_3A.ipynb")
