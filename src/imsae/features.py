"""Per-snapshot vibration features.

One IMS snapshot is 1 second of acceleration sampled at 20 kHz (20480 samples).
Each bearing is summarised by 11 numbers: log RMS, kurtosis, crest factor and the
log power in 8 frequency bands. Kurtosis and crest factor react to impulsive
faults, band powers show the energy moving into the bearing resonance range.
"""
import numpy as np

FS = 20_000
BAND_EDGES_HZ = (0, 500, 1000, 2000, 3000, 4000, 5000, 7000, 10000)
FEATURE_NAMES = ["log_rms", "kurtosis", "crest"] + [
    f"band_{lo}_{hi}Hz" for lo, hi in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:])
]
N_FEATURES = len(FEATURE_NAMES)


def channel_features(x, fs=FS):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    kurt = np.mean(x ** 4) / rms ** 4
    crest = np.max(np.abs(x)) / rms
    power = np.abs(np.fft.rfft(x)) ** 2 / len(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    bands = [
        np.log10(power[(freqs >= lo) & (freqs < hi)].sum() + 1e-12)
        for lo, hi in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:])
    ]
    return np.array([np.log10(rms), kurt, crest, *bands])


def bearing_features(snapshot, channels_per_bearing=1, fs=FS):
    """snapshot: (n_samples, n_channels) -> (n_bearings, N_FEATURES).

    For bearings with two accelerometers (IMS test 1) the first channel is used.
    """
    n_bearings = snapshot.shape[1] // channels_per_bearing
    return np.stack(
        [channel_features(snapshot[:, b * channels_per_bearing], fs) for b in range(n_bearings)]
    )
