"""Reader for the real NASA IMS files (one tab separated text file per snapshot).

Expected layout after unzipping the archive from the NASA Prognostics Data
Repository (or a Kaggle mirror):  <root>/{1st_test,2nd_test,3rd_test|4th_test}/<timestamp files>
Each file: 20480 rows, 8 columns (test 1, two accelerometers per bearing) or
4 columns (tests 2 and 3, one per bearing). File names are timestamps
(YYYY.MM.DD.HH.MM.SS) so sorting by name is chronological.

Failure facts from the IMS readme (bearing numbers there are 1 based):
test 1 -> bearing 3 inner race and bearing 4 roller element,
test 2 -> bearing 1 outer race, test 3 -> bearing 3 outer race.
The readme gives no onset index. FAULT_ONSET_FRACTION below is an ASSUMPTION
used to build labels for threshold tuning and testing; replace it with an onset
you can justify (for example from RMS or kurtosis trends in the literature).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .features import bearing_features

IMS_TESTS = {
    "1st_test": dict(channels_per_bearing=2, faulty=(2, 3)),
    "2nd_test": dict(channels_per_bearing=1, faulty=(0,)),
    "3rd_test": dict(channels_per_bearing=1, faulty=(2,)),
}
FAULT_ONSET_FRACTION = {"1st_test": 0.75, "2nd_test": 0.75, "3rd_test": 0.75}  # ASSUMED


def list_snapshot_files(folder):
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and not p.name.startswith("."))


def read_snapshot(path):
    return pd.read_csv(path, sep=r"\s+", header=None, engine="c").to_numpy(dtype=float)


def folder_features(folder, channels_per_bearing, limit=None):
    """Feature array (n_bearings, n_files, N_FEATURES) for one IMS test folder."""
    files = list_snapshot_files(folder)[:limit]
    return np.stack(
        [bearing_features(read_snapshot(f), channels_per_bearing) for f in files], axis=1
    )
