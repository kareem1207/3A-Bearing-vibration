import numpy as np

from imsae import evaluate as ev
from imsae.features import N_FEATURES, bearing_features, channel_features
from imsae.ims_io import folder_features, read_snapshot
from imsae.models import make_windows
from imsae.simulate import RunSimulator, SimConfig


def test_alarm_rule_debounce():
    s = np.array([0, 5, 0, 5, 5, 0.0])
    assert ev.alarm_flags(s, 1, 1).tolist() == [0, 1, 0, 1, 1, 0]
    assert ev.alarm_flags(s, 1, 2).tolist() == [0, 0, 0, 0, 1, 0]
    assert ev.count_episodes(ev.alarm_flags(s, 1, 1)) == 2


def test_evaluate_counts_and_cost():
    scores = np.array([0, 0, 9, 0, 0, 9, 9, 9, 0, 0.0])   # false alarm at 2, fault from 5, detected at once
    r = ev.evaluate([(scores, 5)], 1.0, 1)
    assert r["fa"] == 1 and r["detected"] == 1 and r["missed"] == 0
    assert r["cost"] == ev.COST_FALSE_ALARM
    assert np.isclose(r["recall"], 3 / 5)
    late = ev.evaluate([(np.r_[np.zeros(10), np.zeros(6), 9, 9.0], 10)], 1.0, 1)  # first alarm 6 readings late
    assert late["missed"] == 1 and late["cost"] == ev.COST_MISS


def test_healthy_series_has_no_fault_labels():
    r = ev.evaluate([(np.array([0, 9, 0.0]), None)], 1.0, 1)
    assert r["events"] == 0 and r["fa"] == 1


def test_windows_are_causal_and_padded():
    X = np.arange(10, dtype=float).reshape(5, 2)
    W = make_windows(X, 3)
    assert W.shape == (5, 3, 2)
    assert np.array_equal(W[0], np.repeat(X[:1], 3, 0))      # padded with first reading
    assert np.array_equal(W[4, -1], X[4]) and np.array_equal(W[4, 0], X[2])


def test_features_react_to_impulses():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20480)
    y = x.copy(); y[::85] += 8
    assert channel_features(y)[1] > channel_features(x)[1] + 1      # kurtosis
    assert len(channel_features(x)) == N_FEATURES


def test_simulator_fault_visible_after_onset_and_deterministic():
    cfg = SimConfig(n_files=40)
    a, b = RunSimulator(cfg, 7), RunSimulator(cfg, 7)
    a.onset = b.onset = 20; a.growth = b.growth = np.log(40) / 20
    Fa, Fb = a.features(), b.features()
    assert np.allclose(Fa, Fb)                                     # fixed seed -> identical data
    f = Fa[a.faulty]
    assert f[-1, 0] > f[:10, 0].mean() + 0.15                       # log RMS rises at the end of the run


def test_ims_reader_roundtrip(tmp_path):
    sim = RunSimulator(SimConfig(n_files=3), 3)
    snaps = [sim.snapshot(k) for k in range(3)]
    d = tmp_path / "2nd_test"; d.mkdir()
    for k, s in enumerate(snaps):                                   # write in the real IMS text format
        np.savetxt(d / f"2004.02.12.10.{32 + k * 10}.39", s, delimiter="\t", fmt="%.5f")
    assert read_snapshot(d / "2004.02.12.10.32.39").shape == (20480, 4)
    F = folder_features(d, 1)
    assert F.shape == (4, 3, N_FEATURES)
    assert np.allclose(F[:, 0], bearing_features(snaps[0]), atol=1e-2)
