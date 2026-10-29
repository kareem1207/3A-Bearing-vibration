"""Physics-inspired simulator of an IMS style run-to-failure test.

WHY THIS EXISTS: the real NASA IMS archive could not be downloaded in the
environment this project was built in. The simulator writes snapshots with the
same shape as the real files (20480 x 4, 20 kHz, one snapshot every 10 minutes)
so that the whole pipeline can be exercised end to end. Results obtained on it
describe the pipeline, NOT real bearings. See README.

Healthy bearing : Gaussian noise + shaft harmonics (2000 rpm = 33.3 Hz), a slow
                  daily amplitude cycle and rare benign transient bursts that
                  look a little like faults (false alarm bait).
Faulty bearing  : from the onset reading an outer-race defect adds impulses at
                  the ball pass frequency (236.4 Hz) that ring a 3.5 kHz
                  resonance; the amplitude grows exponentially until the end.
"""
from dataclasses import dataclass

import numpy as np

from .features import FS, bearing_features


@dataclass
class SimConfig:
    n_files: int = 700
    n_samples: int = 20480
    n_bearings: int = 4
    onset_range: tuple = (0.60, 0.85)
    a0: float = 0.25             # initial impulse amplitude, in units of noise sigma (subtle incipient fault)
    growth_ratio: float = 40.0   # impulse amplitude at the end / at onset
    burst_prob: float = 0.012    # benign transient per snapshot (about 1.7 per day)
    daily_period: int = 144      # snapshots per day at 10 minute spacing
    shaft_hz: float = 33.3
    bpfo_hz: float = 236.4
    resonance_hz: float = 3500.0
    ring_tau_s: float = 5e-4


class RunSimulator:
    def __init__(self, cfg: SimConfig, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        r = self.rng
        self.faulty = int(r.integers(cfg.n_bearings))
        self.onset = int(r.uniform(*cfg.onset_range) * cfg.n_files)
        self.sigma = r.uniform(0.05, 0.09, cfg.n_bearings)
        self.phase = r.uniform(0, 2 * np.pi, cfg.n_bearings)
        ring_n = int(5 * cfg.ring_tau_s * FS)
        n = np.arange(ring_n)
        self.kernel = np.exp(-n / (cfg.ring_tau_s * FS)) * np.sin(2 * np.pi * cfg.resonance_hz * n / FS)
        self.growth = np.log(cfg.growth_ratio) / max(cfg.n_files - self.onset, 1)

    def _add_impulses(self, x, positions, amps):
        L = len(self.kernel)
        pos = positions[positions < len(x) - L]
        a = amps[: len(pos)]
        idx = pos[:, None] + np.arange(L)[None, :]
        np.add.at(x, idx, a[:, None] * self.kernel[None, :])

    def snapshot(self, k):
        c, r = self.cfg, self.rng
        t = np.arange(c.n_samples) / FS
        x = np.zeros((c.n_samples, c.n_bearings))
        for b in range(c.n_bearings):
            drift = 1 + 0.06 * np.sin(2 * np.pi * k / c.daily_period + self.phase[b])
            s = self.sigma[b] * drift * (1 + 0.02 * r.standard_normal())
            x[:, b] = (
                s * r.standard_normal(c.n_samples)
                + 0.30 * s * np.sin(2 * np.pi * c.shaft_hz * t + self.phase[b])
                + 0.15 * s * np.sin(4 * np.pi * c.shaft_hz * t)
            )
        if r.random() < c.burst_prob:                       # benign transient
            b = int(r.integers(c.n_bearings))
            pos = r.integers(0, c.n_samples - 100, int(r.integers(3, 7)))
            self._add_impulses(x[:, b], pos, r.uniform(5, 9, len(pos)) * self.sigma[b])
        if k >= self.onset:                                  # bearing defect
            b = self.faulty
            period = FS / c.bpfo_hz
            n_imp = int(c.n_samples / period)
            pos = ((np.arange(n_imp) + r.random()) * period + r.normal(0, 0.02 * period, n_imp)).astype(int)
            pos = np.clip(pos, 0, c.n_samples - 1)
            amp = c.a0 * self.sigma[b] * np.exp(self.growth * (k - self.onset)) * (1 + 0.3 * r.standard_normal(n_imp))
            self._add_impulses(x[:, b], pos, amp)
            for nb in (b - 1, b + 1):                        # weak leakage to neighbours
                if 0 <= nb < c.n_bearings:
                    self._add_impulses(x[:, nb], pos, 0.08 * amp)
        return x

    def features(self):
        """Feature array of shape (n_bearings, n_files, N_FEATURES)."""
        return np.stack(
            [bearing_features(self.snapshot(k)) for k in range(self.cfg.n_files)], axis=1
        )
