"""Build report/DL_T03_3A.html and docs/plant_engineers_note.html from results/*.csv.
PDFs are produced by scripts/html_to_pdf.js (Chromium)."""
import base64
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1]
res = R / "results"
cmp_ = pd.read_csv(res / "model_comparison_per_seed.csv")
bn = pd.read_csv(res / "bottleneck_sweep.csv")
sq = pd.read_csv(res / "sequence_length.csv")
cost = pd.read_csv(res / "cost_analysis.csv")
rt = pd.read_csv(res / "recall_target_point_per_seed.csv")
sens = pd.read_csv(res / "sensitivity_a0.csv")


def img(name, width):
    b = base64.b64encode((res / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" style="width:{width}">'


def ms(g, col, nd=3):
    return f"{g[col].mean():.{nd}f} ± {g[col].std():.{nd}f}"


order = ["dense_wide", "dense_narrow", "dense_dae_sparse", "lstm_dae"]
rows = ""
for m in order:
    g = cmp_[cmp_.model == m]
    rows += (f"<tr><td>{m}</td><td>{ms(g,'auc')}</td><td>{ms(g,'f1_pt_recall',2)}</td><td>{ms(g,'f1_pt_fa_per_day',2)}</td>"
             f"<td>{ms(g,'f1_pt_f1',2)}</td><td>{g.f1_pt_mean_delay.mean():.0f}</td><td>{g.f1_pt_detected.sum():.0f}/{g.f1_pt_events.sum():.0f}</td>"
             f"<td>{ms(g,'cost_pt_recall',2)}</td><td>{g.cost_pt_fa_per_day.mean():.2f}</td><td>{g.cost_pt_mean_delay.mean():.0f}</td></tr>")
tab_models = ("<table><tr><th rowspan=2>Model</th><th rowspan=2>ROC-AUC</th><th colspan=5>Best-F1 threshold (validation)</th>"
              "<th colspan=3>Minimum-cost threshold (validation)</th></tr><tr><th>recall</th><th>FA/day</th><th>F1</th><th>delay</th><th>in time</th>"
              "<th>recall</th><th>FA/day</th><th>delay</th></tr>" + rows + "</table>")

bn_rows = "".join(f"<tr><td>{'%d (= input)' % r.bottleneck if r.bottleneck == 11 else int(r.bottleneck)}</td><td>{r.auc:.3f}</td>"
                  f"<td>{r.train_loss:.4f}</td><td>{r.err_healthy:.4f}</td></tr>" for r in bn.itertuples())
tab_bn = "<table><tr><th>bottleneck</th><th>AUC</th><th>train MSE</th><th>healthy MSE</th></tr>" + bn_rows + "</table>"

sq_rows = "".join(f"<tr><td>{int(r.T)}</td><td>{r.auc:.3f}</td></tr>" for r in sq.itertuples())
tab_sq = "<table><tr><th>T</th><th>AUC</th></tr>" + sq_rows + "</table>"

c = cost.set_index("operating point")
def cc(col): return [c.loc[k, col] for k in c.index]
tab_cost = ("<table><tr><th></th><th>Best F1 (val.)</th><th>Min cost (val.)</th><th>Recall 0.85 (val.)*</th></tr>"
            f"<tr><td>threshold (error)</td><td>{cc('threshold')[0]:.3f}</td><td>{cc('threshold')[1]:.2f}</td><td>≈0.01</td></tr>"
            f"<tr><td>recall / precision</td><td>{cc('recall')[0]:.2f} / {cc('precision')[0]:.2f}</td><td>{cc('recall')[1]:.2f} / {cc('precision')[1]:.2f}</td><td>0.83–0.85 / 0.11</td></tr>"
            f"<tr><td>false alarms per day</td><td>{cc('false alarms/day')[0]:.2f}</td><td>{cc('false alarms/day')[1]:.2f}</td><td>30.6–33.5</td></tr>"
            f"<tr><td>missed faults (never alarmed)</td><td>{cc('missed faults')[0]}</td><td>{cc('missed faults')[1]}</td><td>0</td></tr>"
            f"<tr><td>alarmed within 3 readings</td><td>{cc('faults detected in time')[0]}</td><td>{cc('faults detected in time')[1]}</td><td>2–3 of 3</td></tr>"
            f"<tr><td>inspection cost (Rs)</td><td>{cc('inspection cost Rs')[0]:,.0f}</td><td>{cc('inspection cost Rs')[1]:,.0f}</td><td>≈34,00,000–37,00,000 total</td></tr></table>"
            "<div class=cap>*3 seeds, dense_dae_sparse; the other models reached recall 0.85 on validation in 1 of 3 (narrow) or 0 of 3 (LSTM) seeds.</div>")

s = sens.pivot(index="a0", columns="model", values="auc").round(3)
tab_sens = ("<table><tr><th>a0 (σ)</th>" + "".join(f"<th>{m}</th>" for m in s.columns) + "</tr>" +
            "".join(f"<tr><td>{i}</td>" + "".join(f"<td>{v:.3f}</td>" for v in row) + "</tr>" for i, row in s.iterrows()) + "</table>")

CSS = """<style>
@page{size:A4;margin:1.5cm 1.6cm}
body{font-family:'DejaVu Serif',Georgia,serif;font-size:12pt;line-height:1.15;color:#111}
h1{font-size:15pt;margin:0 0 2pt}h2{font-size:12.5pt;margin:7pt 0 2pt;color:#1f3a68}p{margin:2pt 0 3pt;text-align:justify}
table{border-collapse:collapse;font-size:9.5pt;margin:3pt 0;width:100%}td,th{border:1px solid #999;padding:1.5pt 3pt;text-align:center}th{background:#e6ecf5}
.cap{font-size:9.5pt;color:#333;margin-bottom:3pt}.warn{border:1.5px solid #b00;padding:3pt 5pt;margin:4pt 0;background:#fff5f5}
ul{margin:2pt 0 2pt 14pt;padding:0}li{margin:0}.two{display:flex;gap:6pt}.two>div{flex:1}
</style>"""

report = f"""<!doctype html><html><head><meta charset=utf-8><title>DL_T03_3A</title>{CSS}</head><body>
<h1>DL_T03_3A: Anomaly Detection in Bearing Vibration Using Autoencoders</h1>
<p><b>Complex Engineering Problem Task 3, variant 3A (NASA IMS bearing vibration), Team 3.</b> CM701PC / CM751PC Deep Learning, MGIT, B.Tech CSE (AI&amp;ML) VII sem.</p>
<table><tr><th>Roll number</th><th>Role (department allotment)</th><th>Main contribution</th></tr>
<tr><td>23261A6660</td><td>Data lead</td><td>Data pipeline, features, simulator, data card, report sections 2–3</td></tr>
<tr><td>23261A6650</td><td>Model lead</td><td>Autoencoder models, training, README, section 4</td></tr>
<tr><td>23261A6630</td><td>Evaluation and report lead</td><td>Metrics, plots, target check, sections 5–7</td></tr></table>
<div class=cap>Roles follow the allotment order; the team must confirm the actual split before submission.</div>
<div class=warn><b>Data status (read first).</b> The NASA IMS archive could not be downloaded in the environment where this project was built. All results below come from a <b>simulated</b> IMS style dataset (same file shape, 20 kHz, 10 minute spacing) and describe the <b>pipeline</b>, not real bearings. The real-data loader exists and is tested on files in the IMS format, but has not been run on the real archive.</div>

<h2>1. Problem understanding</h2>
<p>Breakdowns are rare, so a warning system must learn <i>normal</i> behaviour only and flag departures. The targets pull against each other: recall ≥ 0.85 on fault readings pushes the alarm threshold down; fewer than 2 false alarms per 24 h pushes it up; detection must come within 3 readings (the task's rule for data sampled slower than once a minute, here 10 minutes); no labelled fault may be used for training. A missed fault costs Rs 4 lakh and a false alarm Rs 2,000, a 200:1 ratio. Constraints I identified: only an unsupervised model is allowed; fault onset is not labelled in the real data, so any labels are an assumption; with one fault per run, test statistics rest on few events (3 test faults here).</p>

<h2>2. Data preparation and exploratory analysis</h2>
<p><b>Simulator</b> (<tt>src/imsae/simulate.py</tt>): 8 run-to-failure runs × 4 bearings (700 snapshots each, 20480 samples, 4.9 days). Healthy bearings: noise, shaft harmonics at 33.3 Hz, a 6% daily amplitude cycle and benign transient bursts (≈1.7 per day) that imitate faults. One bearing per run develops an outer-race defect: impulses at 236.4 Hz ringing a 3.5 kHz resonance, amplitude 0.25σ at onset growing 40×. Split by <b>run</b>: 3 train, 2 validation, 3 test (seed-fixed). <b>Features</b>: per bearing 11 numbers per snapshot (log RMS, kurtosis, crest factor, 8 band log-powers). Each series is standardised with its own first 40% (assumed healthy; no labels). Training uses only that prefix (last 20% of it for early stopping). Validation runs choose thresholds; test runs give every reported number. Class balance: about 5–10% of readings are faulty. Fig. 2 shows the key EDA finding: the error of a healthy-trained model stays flat until well after the labelled onset.</p>

<h2>3. Models and experiments</h2>
<p>All are Keras autoencoders trained on healthy readings (Adam 1e-3, batch 32, ≤100 epochs, early stopping, 3 seeds 0–2). Score = reconstruction MSE of the newest reading. <b>dense_wide</b>: 11→64→11→64→11 (bottleneck = input). <b>dense_narrow</b>: bottleneck 8. <b>dense_dae_sparse</b>: bottleneck 16, input noise 0.2, L1 activity penalty 1e-3 (noise robustness and sparsity). <b>lstm_dae</b> (my "improved" model): LSTM encoder/decoder over a causal window of T=6 readings, bottleneck 8, input noise 0.1. The alarm fires when the score exceeds a threshold for m consecutive readings (m ∈ {1,2,3} and the threshold are chosen on validation data only). Two thresholds are compared: best reading-level F1, and minimum total cost (Rs 4 lakh per <i>missed</i> fault + Rs 2,000 per false alarm episode).</p>
<p><b>Cost model change (disclosed).</b> My first version counted an alarm later than 3 readings as a miss. Every model then scored "missed", so "never alarm" became cheapest, which is meaningless. I changed it after seeing those first results: a fault is <i>missed</i> only if never alarmed before the run ends; timeliness is reported separately as "in time" (alarm within 3 readings). All tables use the corrected model.</p>
{tab_models}
<div class=cap>Table 1. Test results, mean ± std over 3 seeds (3 test fault events). FA/day = false alarm episodes per 24 h on healthy data; delay = mean readings from onset to first alarm (1 reading = 10 min); in time = faults alarmed within 3 readings, summed over seeds (3 events × 3 seeds).</div>

<h2>4. Results against every target</h2>
<table><tr><th>Target</th><th>Result (best model, test)</th><th>Met?</th></tr>
<tr><td>Recall on fault readings ≥ 0.85</td><td>0.65 at best-F1 point (dense_dae_sparse); 0.83–0.85 only at ≈ 33 FA/day</td><td><b>No</b></td></tr>
<tr><td>False alarms &lt; 2 per 24 h</td><td>0.6–1.0 at best-F1; 0.0 at min-cost point; ≈ 33 when recall 0.85 is forced</td><td>Yes, but not together with recall</td></tr>
<tr><td>Detection within 3 readings</td><td>0 of 3 faults at best-F1 and min-cost points (mean delay 45 and 108 readings); 2–3 of 3 when recall 0.85 is forced</td><td><b>No</b> at the usable points</td></tr>
<tr><td>No labelled faults in training</td><td>Training uses only the first 40% of each series; labels only for thresholds and testing</td><td>Yes</td></tr></table>
<div class=two><div>{img('fig_pr_curve.png','100%')}<div class=cap>Fig. 1. Precision-recall curve (test, dense_dae_sparse, seed 0) with both operating points. Even the best curve never reaches recall 0.85 above precision ≈ 0.1.</div></div>
<div>{img('fig_timeline.png','100%')}<div class=cap>Fig. 2. Score timeline of a faulty bearing: the score leaves the healthy band only ≈ 40 readings after the labelled onset.</div></div></div>
{tab_cost}
<div class=cap>Table 2. Cost analysis, dense_dae_sparse seed 0 on the 3 test faults (about 55 healthy days).</div>

<h2>5. Answers to the report questions</h2>
<p><b>Q1. Why can an autoencoder with a bottleneck as wide as its input fail?</b> With no compression a network with a hidden layer wider than the input can represent the identity map: capacity exceeds what the healthy data needs, so the training error becomes tiny ({bn.train_loss.iloc[3]:.4f} versus {bn.train_loss.iloc[2]:.4f} at width 8 and {bn.train_loss.iloc[0]:.3f} at width 2) and it also reconstructs mildly abnormal inputs well; this is the overfitting/low-bias end of the capacity curve, and the model has learned nothing about what normal <i>looks like</i>. A narrow bottleneck forces the network to keep only the structure shared by healthy data, so deviations from it cost error (the underfitting end appears at width 2, where even healthy error rises to {bn.err_healthy.iloc[0]:.3f}). Evidence from my runs: AUC is {bn.auc.iloc[3]:.3f} for the input-width bottleneck versus {bn.auc.iloc[1]:.3f} at width 4 and {bn.auc.iloc[2]:.3f} at 8 (Table 3). The gap is modest (about 1–2 seed standard deviations of the wide model, 0.02) and AUC is not monotone, so this supports but does not prove the argument. {tab_bn}<div class=cap>Table 3. Dense AE bottleneck sweep (3 seeds, test).</div></p>
<p><b>Q2. Did the improved model beat the dense baseline?</b> No. lstm_dae reaches AUC {ms(cmp_[cmp_.model=='lstm_dae'],'auc')} versus {ms(cmp_[cmp_.model=='dense_narrow'],'auc')} for dense_narrow, and the T ablation (T = 1, 3, 6, 12 → AUC {', '.join(f'{v:.3f}' for v in sq.auc)}) shows no gain from longer context. Reasons I can support: each reading's 11 features already summarise a full second of vibration, and in this simulator the fault is stationary within a reading and grows smoothly, so what a recurrent model adds (temporal dependence between readings) carries little extra signal; the LSTM has more parameters yet the same 700-reading healthy training set, and it never triggered early stopping before the 100-epoch cap, so it may be under-trained, which I did not test. The best AUC came from the denoising, sparse dense model ({ms(cmp_[cmp_.model=='dense_dae_sparse'],'auc')}), a small gain (≈ 2 standard deviations) from noise robustness and sparsity. On real IMS data temporal context may matter more; I cannot say.</p>
<p><b>Q3. Which threshold, given Rs 4 lakh per miss and Rs 2,000 per false alarm?</b> Break-even is 200 false alarms per missed fault, so one expects the cost-optimal threshold to sit <i>below</i> the best-F1 one. Here it sits far <i>above</i> it ({cost.threshold.iloc[1]:.2f} versus {cost.threshold.iloc[0]:.3f}), because the simulated defect eventually grows so large that even a high threshold catches every fault before the run ends with zero false alarms (Rs 0 on the test runs versus Rs {cost['total cost Rs'].iloc[0]:,.0f} for best-F1). The cost model, however, only counts breakdowns and false alarms; it ignores how <i>late</i> the warning comes. The min-cost threshold alarms on average 108 readings (≈ 18 h) after onset, the best-F1 one 45 readings (≈ 7.5 h). Because maintenance needs lead time I <b>recommend the best-F1 threshold</b> (error ≈ 0.046, m = 1): about one false alarm a day (inside the limit) and warning ≈ 10 h earlier, for roughly Rs 2,000 a day. Forcing recall to 0.85 needs ≈ 33 false alarms a day (Rs 66,000 a day), far outside the limit; the task's recall and false-alarm targets cannot both be met on this data.</p>

<h2>6. Limitations and next steps</h2>
<p>(i) <b>Simulated data</b>: absolute numbers are properties of my simulator (onset amplitude 0.25σ, 40× growth). A sensitivity run over the onset amplitude (AUC, Table 4) shows the picture depends strongly on it and that the model ranking is stable but small. Real IMS has no onset labels, load changes and sensor drift. (ii) Only 3 test faults and 3 seeds: differences of 0.01–0.03 AUC are within noise. (iii) The "in time" target is unattainable when the labelled onset is below the noise floor; a defensible onset label (for example from RMS or kurtosis trends) is needed for real data. (iv) The cost model changed after first results (section 3). (v) With more time: run on real IMS with literature onset labels, per-bearing thresholds, trend (CUSUM) alarms on the score, and train the LSTM longer.</p>
{tab_sens}<div class=cap>Table 4. Test ROC-AUC versus initial defect amplitude a0 (2 seeds): the easier the fault, the higher every model's AUC.</div>

<h2>7. References and AI-use declaration</h2>
<p style="font-size:10.5pt">[1] J. Lee, H. Qiu, G. Yu, J. Lin, "Rexnord Technical Services: Bearing Data Set," IMS, Univ. of Cincinnati, NASA Prognostics Data Repository, 2007. [2] H. Qiu, J. Lee, J. Lin, G. Yu, "Wavelet filter-based weak signature detection method and its application on rolling element bearing prognostics," <i>J. Sound Vib.</i>, vol. 289, pp. 1066–1090, 2006. [3] I. Goodfellow, Y. Bengio, A. Courville, <i>Deep Learning</i>, MIT Press, 2016, ch. 5, 14. [4] D. P. Kingma, J. Ba, "Adam: A method for stochastic optimization," in <i>Proc. ICLR</i>, 2015. [5] F. Chollet et al., "Keras," 2015.<br>
<b>Appendix: AI use.</b> The code, experiment design and first draft of this report were produced with Claude Code (Anthropic AI coding assistant) in a session with the data lead. All numbers come from the code in the repository and can be regenerated with <tt>python scripts/run_experiments.py</tt>, <tt>scripts/sensitivity_a0.py</tt> and <tt>scripts/recall_target_point.py</tt>. The assistant could not access the real dataset, so it built the simulator; it also proposed the cost-model correction in section 3. Team members must read and be able to explain every line before the viva.</p>
</body></html>"""
(R / "report").mkdir(exist_ok=True)
(R / "report" / "DL_T03_3A.html").write_text(report)

note = f"""<!doctype html><html><head><meta charset=utf-8><title>Note for plant engineers</title>{CSS}</head><body>
<h1>Note for plant engineers: the bearing early warning system</h1>
<p><b>Status: tested on simulated data only.</b> The numbers below show how the method behaves on a computer-made test, not on your machines. It must be re-checked on real recordings before anyone relies on it.</p>
<h2>What it does</h2>
<p>Every ten minutes the system listens to one second of vibration from each bearing and compares it with what that bearing sounds like when healthy. A healthy bearing is "learned" from its first days of running; no failed bearings are needed. If the vibration drifts away from the learned pattern, the system raises an alarm.</p>
<h2>What we found</h2>
<ul>
<li>A failing bearing was always caught eventually, but <b>not early</b>. At the setting we recommend, the alarm came on average about <b>7 hours</b> after the fault began (the early signs are smaller than normal noise).</li>
<li>At the recommended setting there is about <b>one false alarm a day</b>. A false alarm means an unnecessary inspection (about Rs 2,000).</li>
<li>Only about <b>6 in 10</b> of the abnormal readings were flagged at that setting. We could not reach our goal of 85 in 10 without raising roughly <b>33 false alarms a day</b>, which is far too many; you would switch it off.</li>
<li>A very quiet setting (no false alarms at all) still caught every fault in the end, but only about <b>18 hours</b> after it began, when it was already well advanced.</li>
</ul>
<h2>What we recommend</h2>
<p>Use the middle setting: about one false alarm per day, warning several hours before a fault becomes severe. Treat an alarm as "go and look at this bearing today", not as "the bearing is about to fail". A missed breakdown costs about Rs 4 lakh, a needless inspection Rs 2,000, so inspecting even a hundred alarms is cheaper than one breakdown; but too many false alarms make people ignore the system, which is why the limit of two a day matters.</p>
<h2>What we are not sure about</h2>
<ul><li>Real machines have load changes, sensor ageing and dirt that our test data does not copy. The false alarm rate could be higher.</li>
<li>The test had only three failures, so the figures are rough.</li>
<li>Nobody has labelled the exact moment the real bearings started to fail; the timing figures depend on that.</li></ul>
<h2>What we need from you</h2>
<p>Recordings from one healthy week of each machine, and the maintenance log of past failures with dates, so we can check the system against real events.</p>
</body></html>"""
(R / "docs" / "plant_engineers_note.html").write_text(note)
print("built")
