"""Forecast features and targets, shared by the evaluation and the live replay.

`origin_features(cal, positions)` reads only grid slots at or before each origin, the
calibration state, and inputs from events logged at or before the origin. Targets come from a
separate function, `targets`, so future truth is never in the same code path as features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import Calibration
from .model import GLUCOSE_CEILING, GLUCOSE_FLOOR, simulate_array
from .protocol import GRID_MIN, HORIZONS_MIN, LAGS_MIN, TREND_WINDOW_MIN

DYNAMIC = ["g0", "lag15", "lag30", "lag60", "lag120", "sd_2h", "hour_sin", "hour_cos", "insulin_now", "iob",
           "su_now", "glp1_now", "carbs_now", "cob", "offset"]
EHR = ["age", "bmi", "hba1c", "egfr", "duration_years", "c_peptide", "male", "on_su", "on_insulin"]
PHYSIOLOGY = ["twin_30", "twin_60", "twin_120", "twin_min_120"]
ARMS = {"A": DYNAMIC, "B": DYNAMIC + EHR, "D": DYNAMIC + PHYSIOLOGY, "C": DYNAMIC + EHR + PHYSIOLOGY}
ARM_NAMES = {"A": "dynamic only (CGM history, logged doses and meals, time)",
             "B": "dynamic + clinical record (machine learning only)",
             "D": "dynamic + physiology core",
             "C": "hybrid: dynamic + clinical record + physiology core"}


def eligible_origins(cal: Calibration, upto: int | None = None) -> np.ndarray:
    """Grid positions after calibration with a reading now and 15 minutes earlier."""
    n_obs = len(cal.ref) if upto is None else min(upto + 1, len(cal.ref))
    start = max(cal.cal_end_pos, 1)
    if start >= n_obs:
        return np.array([], dtype=int)
    pos = np.arange(start, n_obs)
    ok = np.isfinite(cal.ref[pos]) & np.isfinite(cal.ref[pos - 1])
    return pos[ok]


def _history(ref: np.ndarray, i: int, back_steps: int) -> np.ndarray:
    lo = max(i - back_steps, 0)
    return ref[lo:i + 1]


def trend_line(ref: np.ndarray, i: int) -> tuple[float, float] | None:
    """Least-squares line (intercept at the origin, slope per minute) through the readings in the
    last 30 minutes; None if fewer than 3 readings."""
    steps = TREND_WINDOW_MIN // GRID_MIN
    ys = ref[i - steps:i + 1] if i >= steps else ref[:i + 1]
    xs = np.arange(-(len(ys) - 1), 1, dtype=float) * GRID_MIN
    ok = np.isfinite(ys)
    if ok.sum() < 3:
        return None
    x, y = xs[ok], ys[ok]
    b = float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())
    return float(y.mean() - b * x.mean()), b


def trend_forecast(ref: np.ndarray, i: int, horizon_min: float) -> float:
    """The trend line extrapolated `horizon_min` ahead; persistence when it is undefined."""
    line = trend_line(ref, i)
    if line is None:
        return float(ref[i])
    a, b = line
    return float(np.clip(a + b * horizon_min, GLUCOSE_FLOOR, GLUCOSE_CEILING))


def origin_features(cal: Calibration, positions: np.ndarray) -> pd.DataFrame:
    """One row per origin. Reads cal.ref only at or before each origin."""
    ref, rows = cal.ref, []
    h_steps = max(HORIZONS_MIN) // GRID_MIN
    for i in np.asarray(positions, dtype=int):
        t = cal.grid[i]
        g0 = float(ref[i])
        row = {"period": cal.period_id, "patient": cal.patient_id, "pos": int(i), "time": t, "g0": g0}
        for lag in LAGS_MIN:
            j = i - lag // GRID_MIN
            row[f"lag{lag}"] = float(ref[j]) if j >= 0 else np.nan
        hist = _history(ref, i, 120 // GRID_MIN)
        hist = hist[np.isfinite(hist)]
        row["sd_2h"] = float(np.std(hist)) if len(hist) >= 3 else np.nan
        hour = t.hour + t.minute / 60.0
        row["hour_sin"], row["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        ins, iob, su, glp, carbs, cob = cal.inputs_now[i]
        row.update(insulin_now=ins, iob=iob, su_now=su, glp1_now=glp, carbs_now=carbs, cob=cob, offset=cal.offset)
        row.update(cal.static)
        idx = np.arange(i, i + h_steps + 1)
        sim = simulate_array(cal.twin, g0, cal.bank.known_array(i, idx), np.full(len(idx), float(GRID_MIN)))
        line = trend_line(ref, i)
        for h in HORIZONS_MIN:
            row[f"twin_{h}"] = float(sim[h // GRID_MIN] - g0)
            row[f"trend_{h}"] = g0 if line is None else float(np.clip(line[0] + line[1] * h, GLUCOSE_FLOOR,
                                                                         GLUCOSE_CEILING))
        row["twin_min_120"] = float(sim.min() - g0)
        rows.append(row)
    return pd.DataFrame(rows)


def targets(cal: Calibration, positions: np.ndarray) -> pd.DataFrame:
    """Future truth for scoring only: the reference at exactly t + h on the grid."""
    out = {}
    for h in HORIZONS_MIN:
        j = np.asarray(positions, dtype=int) + h // GRID_MIN
        vals = np.where(j < len(cal.ref), cal.ref[np.minimum(j, len(cal.ref) - 1)], np.nan)
        out[f"y_{h}"] = vals - cal.ref[np.asarray(positions, dtype=int)]
    return pd.DataFrame(out)
