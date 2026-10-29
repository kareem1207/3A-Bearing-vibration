"""Data splits and experiment drivers."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .evaluate import choose_operating_points, evaluate, pooled_auc
from .models import Detector
from .simulate import RunSimulator, SimConfig

SEEDS = (0, 1, 2)
PREFIX_FRAC = 0.40   # leading share of every series assumed healthy: used for training and per-series calibration
MODELS = {
    "dense_wide": dict(kind="dense", bottleneck="input", hidden=64),
    "dense_narrow": dict(kind="dense", bottleneck=8, hidden=64),
    "dense_dae_sparse": dict(kind="dense", bottleneck=16, hidden=64, noise=0.2, l1=1e-3),
    "lstm_dae": dict(kind="lstm", T=6, units=32, bottleneck=8, noise=0.1),
}


@dataclass
class Series:
    name: str
    X: np.ndarray            # (n_readings, n_features), standardised
    onset: "int | None"      # first faulty reading, None for a healthy bearing


def calibrate(X, prefix_frac):
    """Standardise a series with the mean/std of its own healthy prefix (no labels used)."""
    p = X[: max(int(len(X) * prefix_frac), 2)]
    return (X - p.mean(0)) / np.maximum(p.std(0), 1e-3)


def sim_split(n_train=3, n_val=2, n_test=3, cfg=None, base_seed=1000, cache=None):
    cfg = cfg or SimConfig()
    out = {"train": [], "val": [], "test": []}
    plan = [("train", n_train), ("val", n_val), ("test", n_test)]
    run_id = 0
    for split, n in plan:
        for _ in range(n):
            path = Path(cache) / f"sim_{base_seed + run_id}_{cfg.n_files}.npz" if cache else None
            if path is not None and path.exists():
                z = np.load(path)
                F, faulty, onset = z["F"], int(z["faulty"]), int(z["onset"])
            else:
                sim = RunSimulator(cfg, base_seed + run_id)
                F, faulty, onset = sim.features(), sim.faulty, sim.onset
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, F=F, faulty=faulty, onset=onset)
            for b in range(F.shape[0]):
                out[split].append(Series(f"run{run_id}_b{b + 1}", calibrate(F[b], PREFIX_FRAC),
                                         onset if b == faulty else None))
            run_id += 1
    return out


def ims_split(root, tests_train=("1st_test", "2nd_test", "3rd_test"), val=("2nd_test",),
              test=("1st_test", "3rd_test"), limit=None):
    """Real IMS data; see ims_io for the assumptions behind onset labels."""
    from .ims_io import FAULT_ONSET_FRACTION, IMS_TESTS, folder_features
    feats = {t: folder_features(Path(root) / t, IMS_TESTS[t]["channels_per_bearing"], limit)
             for t in set(tests_train) | set(val) | set(test)}

    def mk(tests):
        res = []
        for t in tests:
            for b in range(feats[t].shape[0]):
                X = calibrate(feats[t][b], 0.30)
                onset = int(FAULT_ONSET_FRACTION[t] * len(X)) if b in IMS_TESTS[t]["faulty"] else None
                res.append(Series(f"{t}_b{b + 1}", X, onset))
        return res
    return {"train": mk(tests_train), "val": mk(val), "test": mk(test)}


def _train_tail(series_list, prefix_frac=PREFIX_FRAC):
    tr, va = [], []
    for s in series_list:
        p = s.X[: int(len(s.X) * prefix_frac)]
        cut = int(len(p) * 0.8)
        tr.append(p[:cut]); va.append(p[cut:])
    return tr, va


def fit_and_score(spec, seed, data, epochs=100):
    d = data["train"][0].X.shape[1]
    det = Detector(spec, d, seed)
    tr, va = _train_tail(data["train"])
    hist = det.fit(tr, va, epochs=epochs)
    scores = {sp: [(det.score(s.X), s.onset, s.name) for s in data[sp]] for sp in ("val", "test")}
    return scores, hist


def _pairs(rows):
    return [(s, o) for s, o, _ in rows]


def test_at_operating_points(scores):
    """Pick thresholds on validation scores, apply them unchanged to test scores."""
    ops = choose_operating_points(_pairs(scores["val"]))
    test = _pairs(scores["test"])
    res = {"m": ops["m"], "auc": pooled_auc(test)}
    for k in ("cost", "f1"):
        res[k] = dict(val=ops[k], test=evaluate(test, ops[k]["thr"], ops["m"]))
    return res, ops
