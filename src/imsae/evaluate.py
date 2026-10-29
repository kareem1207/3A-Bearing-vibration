"""Alarm rule, targets and cost model.

Cost model: a fault is MISSED (Rs 4 lakh) only if no alarm is raised at any
reading from its onset to the end of the run; a late alarm still prevents the
breakdown. Timeliness is a separate target: 'detected' counts faults alarmed
within DELAY_BUDGET readings. (A first version charged late alarms as misses,
which made 'never alarm' the cheapest policy; see report section 4.)

Sampling is one snapshot per 10 minutes, so the 5 minute delay target becomes
'within 3 readings' (the rule the task gives for data sampled less than once a
minute) and 24 hours = 144 readings.
"""
import numpy as np

READINGS_PER_DAY = 144
DELAY_BUDGET = 3
COST_MISS = 400_000     # Rs per missed fault event
COST_FALSE_ALARM = 2_000  # Rs per false alarm


def alarm_flags(scores, thr, m=1):
    """Alarm when the score exceeds thr for m consecutive readings."""
    above = scores > thr
    if m == 1:
        return above
    return np.convolve(above.astype(int), np.ones(m, dtype=int))[: len(above)] >= m


def count_episodes(flags):
    flags = np.asarray(flags, bool)
    return int(flags[0]) + int(np.sum(flags[1:] & ~flags[:-1])) if len(flags) else 0


def evaluate(series, thr, m=1):
    """series: list of (scores, onset or None). Pooled metrics at one operating point."""
    fa = fp = normal_len = tp = fault_len = events = detected = alarmed = 0
    delays = []
    for scores, onset in series:
        f = alarm_flags(scores, thr, m)
        on = len(scores) if onset is None else onset
        fa += count_episodes(f[:on])
        fp += int(f[:on].sum())
        normal_len += on
        if onset is not None:
            events += 1
            fault_len += len(scores) - on
            tp += int(f[on:].sum())
            hit = np.flatnonzero(f[on:])
            if len(hit):
                alarmed += 1
                delays.append(int(hit[0]))
                detected += hit[0] <= DELAY_BUDGET
    recall = tp / fault_len if fault_len else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(
        thr=float(thr), m=m, recall=recall, precision=precision, f1=f1,
        fa=fa, fa_per_day=fa / (normal_len / READINGS_PER_DAY),
        events=events, detected=int(detected), missed=events - alarmed,
        mean_delay=float(np.mean(delays)) if delays else float("nan"),
        cost=COST_MISS * (events - alarmed) + COST_FALSE_ALARM * fa,
    )


def threshold_grid(series, n=300):
    pooled = np.concatenate([s for s, _ in series])
    return np.unique(np.quantile(pooled, np.linspace(0.5, 1.0, n)))


def sweep(series, m, grid=None):
    grid = threshold_grid(series) if grid is None else grid
    return [evaluate(series, t, m) for t in grid]


def _plateau_middle(values, best):
    """Middle of the longest run of thresholds attaining the best value (robust to ties)."""
    idx = np.flatnonzero(np.isclose(values, best))
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    run = max(runs, key=len)
    return int(run[len(run) // 2])


def choose_operating_points(val_series, ms=(1, 2, 3)):
    """Pick (m, threshold) on VALIDATION data. Returns dict with 'cost' and 'f1' points."""
    best = None
    for m in ms:
        rows = sweep(val_series, m)
        c = np.array([r["cost"] for r in rows], float)
        i = _plateau_middle(c, c.min())
        key = (c.min(), m)
        if best is None or key < best[0]:
            best = (key, m, rows, i)
    _, m, rows, i_cost = best
    f = np.array([r["f1"] for r in rows])
    return dict(m=m, cost=rows[i_cost], f1=rows[_plateau_middle(f, f.max())], rows=rows)


def pooled_auc(series):
    """ROC-AUC of raw scores, healthy readings (label 0) vs fault readings (label 1)."""
    from sklearn.metrics import roc_auc_score
    y, s = [], []
    for scores, onset in series:
        on = len(scores) if onset is None else onset
        y += [0] * on + [1] * (len(scores) - on)
        s += list(scores)
    return float(roc_auc_score(y, s))


TARGETS = dict(recall=0.85, fa_per_day=2.0, delay_readings=DELAY_BUDGET)
