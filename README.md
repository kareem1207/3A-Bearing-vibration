# 3A Bearing Vibration: autoencoder early warning (NASA IMS)

**DL_T03_3A** — Complex Engineering Problem Task 3 (anomaly detection in industrial IoT sensor streams using autoencoders), variant **3A Bearing vibration**, Team 3.
CM701PC Deep Learning / CM751PC Deep Learning Lab, Dept. of Emerging Technologies, MGIT (Autonomous), Hyderabad.

| Roll number | Role |
|---|---|
| 23261A6660 | Data lead |
| 23261A6650 | Model lead |
| 23261A6630 | Evaluation and report lead |

> ## Read this first: data status
> The NASA IMS archive **could not be downloaded** in the environment where this project was built (NASA, Kaggle and similar hosts were blocked).
> Every result in this repository comes from a **simulated IMS-style dataset** (`src/imsae/simulate.py`: same snapshot shape, 20 kHz, one snapshot per 10 minutes, outer-race defect on one bearing).
> The results describe the **pipeline and the trade-offs**, **not real bearings**.
> A reader for the real IMS files (`src/imsae/ims_io.py`) is included and unit-tested on files written in the IMS format, and `--data ims` runs the same pipeline on the real archive. It has **not** been run on the real data. IMS has no published fault-onset labels, so the onset used for labels is an assumption you must set (`FAULT_ONSET_FRACTION`).

## The problem
A plant records vibration continuously; breakdowns are rare. Build an **unsupervised** warning system that learns normal behaviour and flags departures, subject to targets that pull against each other:

| Target | Required |
|---|---|
| Recall on labelled fault events | ≥ 0.85 |
| False alarms | < 2 per 24 h of data |
| Detection delay | < 5 min, or within 3 readings for slowly sampled data (here 10 min sampling → 3 readings) |
| Labelled faults used for training | none (labels only for threshold tuning and testing) |

Cost model for the threshold question: a missed fault costs Rs 4 lakh, a false alarm Rs 2,000.

## Approach
1. **Features**: 11 per bearing per 1 s snapshot (log RMS, kurtosis, crest factor, 8 band log-powers).
2. **Calibration/training**: each series is standardised on its own first 40 % (assumed healthy); only that healthy prefix trains the models.
3. **Models** (Keras): `dense_wide` (bottleneck = input width), `dense_narrow` (8), `dense_dae_sparse` (denoising + L1 sparsity), `lstm_dae` (LSTM autoencoder over 6 readings). Score = reconstruction MSE of the newest reading.
4. **Alarm rule**: score > threshold for *m* consecutive readings. Threshold and *m* are chosen on validation runs only; test runs are never used for tuning. Two thresholds are compared: best F1 and minimum cost.
5. Split by *run* (3 train, 2 validation, 3 test), fixed seeds, 3 training seeds.

## Results (simulated data, test runs, mean over 3 seeds)
| Model | ROC-AUC | Best-F1 point: recall | FA/day | faults alarmed within 3 readings | Min-cost point: recall | FA/day |
|---|---|---|---|---|---|---|
| dense_wide | 0.825 | 0.60 | 1.01 | 0/9 | 0.18 | 0.00 |
| dense_narrow | 0.853 | 0.64 | 0.63 | 0/9 | 0.24 | 0.00 |
| dense_dae_sparse | 0.866 | 0.65 | 0.90 | 0/9 | 0.23 | 0.00 |
| lstm_dae | 0.831 | 0.57 | 0.70 | 0/9 | 0.16 | 0.00 |

**Targets are not met, and the report says so.** At the best-F1 threshold, recall is ≈ 0.65 with ≈ 1 false alarm/day and no fault alarmed within 3 readings (mean delay ≈ 7.5 h). Forcing recall to 0.85 needs ≈ 33 false alarms/day. The minimum-cost threshold raises no false alarms and still catches every fault before the run ends, but ≈ 18 h after onset. The "improved" LSTM did **not** beat the dense baseline. See `report/DL_T03_3A.pdf` for the full analysis and answers to the three report questions.

Two things worth knowing about the process (both disclosed in the report): the first cost model counted late alarms as misses, which made "never alarm" optimal, so it was corrected after seeing results; and the simulator's fault amplitude is a design choice, so a sensitivity run over it is included (`results/sensitivity_a0.csv`).

## Repository layout
```
src/imsae/            package: features, simulator, IMS reader, models, evaluation, pipeline
scripts/              run_experiments.py, sensitivity_a0.py, recall_target_point.py, build_report.py, make_notebook.py
notebooks/DL_T03_3A.ipynb   Colab/Jupyter notebook (executed; USE_REAL_IMS switch)
tests/                unit tests (alarm rule, cost, windows, features, simulator, IMS file round-trip)
results/              tables (csv), figures (png), summary.json
docs/                 data_card.md, plant_engineers_note.pdf (one page, non-technical)
report/DL_T03_3A.pdf  3-4 page report (HTML source alongside)
```

## Reproduce
```bash
pip install -r requirements.txt
python -m pytest -q                          # 7 tests
python scripts/run_experiments.py            # about 35 min on 4 CPU cores; tables + figures in results/
python scripts/sensitivity_a0.py             # amplitude sensitivity
python scripts/recall_target_point.py        # cost of forcing recall 0.85
python scripts/run_experiments.py --quick    # 1 minute smoke run
```
Real data: unzip the IMS archive so that `<root>/1st_test`, `2nd_test`, `3rd_test` hold the snapshot files, then
`python scripts/run_experiments.py --data ims --ims-root <root>`. Before trusting the output, review the assumed onset labels in `src/imsae/ims_io.py`.

Seeds are fixed (simulator seeds 1000+, training seeds 0–2); small run-to-run differences can remain because TensorFlow CPU kernels are not forced to be deterministic.

## Licence
MIT, see [LICENSE](LICENSE). The NASA IMS dataset is a separate public dataset with its own terms.

## AI-use declaration
This project was built with Claude Code (Anthropic). See the appendix of the report. Team members must be able to explain every part of the code.

## References
[1] J. Lee, H. Qiu, G. Yu, J. Lin, "Rexnord Technical Services: Bearing Data Set," IMS, Univ. of Cincinnati, NASA Prognostics Data Repository, 2007.
[2] H. Qiu et al., "Wavelet filter-based weak signature detection method and its application on rolling element bearing prognostics," J. Sound Vib., 289, 2006.
[3] I. Goodfellow, Y. Bengio, A. Courville, *Deep Learning*, MIT Press, 2016.
