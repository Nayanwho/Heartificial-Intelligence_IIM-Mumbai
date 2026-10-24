"""Check the sensor before trusting it.

The professional CGM in ShanghaiT2DM reads lower than fingersticks, so a raw
sensor low is often not a real low. The twin therefore:

1. drops each sensor's first 24 hours, when readings are least reliable;
2. learns each person's offset from fingersticks taken within 8 minutes of a
   sensor reading, shrunk towards the cohort median when a person has few pairs;
3. counts a low only after 15 minutes below 70 mg/dL, which at 15-minute sampling
   means two readings in a row.

Version 1 (19 October) learned the offset from the whole period's fingersticks, so
later fingersticks shaped earlier inputs. Version 2 learns it from the calibration
interval only (`calibration_offset`) and freezes it; see docs/evaluation_protocol.md.
`check_sensor` is kept for the descriptive data check, which reports whole-period
statistics and is not used for forecasting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WARMUP_HOURS = 24
PAIR_TOLERANCE_MIN = 8
SHRINK_PAIRS = 10
LOW_MG_DL = 70.0
SEVERE_LOW_MG_DL = 54.0
NIGHT_START_HOUR, NIGHT_END_HOUR = 0, 6
MAX_GAP_MIN = 20


def trim_warmup(cgm: pd.Series, hours: float = WARMUP_HOURS) -> pd.Series:
    if cgm.empty:
        return cgm
    return cgm[cgm.index >= cgm.index.min() + pd.Timedelta(hours=hours)]


def pair_fingersticks(cgm: pd.Series, fingerstick: pd.Series,
                      tolerance_min: float = PAIR_TOLERANCE_MIN) -> pd.DataFrame:
    """Each fingerstick matched to the nearest sensor reading within the tolerance."""
    if cgm.empty or fingerstick.empty:
        return pd.DataFrame(columns=["time", "cgm", "fingerstick", "diff"])
    left = fingerstick.rename("fingerstick").rename_axis("time").reset_index().sort_values("time")
    right = cgm.rename("cgm").rename_axis("time").reset_index().sort_values("time")
    pairs = pd.merge_asof(left, right, on="time", direction="nearest",
                          tolerance=pd.Timedelta(minutes=tolerance_min)).dropna()
    pairs["diff"] = pairs["fingerstick"] - pairs["cgm"]
    return pairs[["time", "cgm", "fingerstick", "diff"]].reset_index(drop=True)


def person_offset(pairs: pd.DataFrame, prior: float, k: float = SHRINK_PAIRS) -> float:
    """Median fingerstick-minus-sensor gap, shrunk towards the cohort prior."""
    n = len(pairs)
    if n == 0:
        return float(prior)
    return float((n * pairs["diff"].median() + k * prior) / (n + k))


@dataclass
class SensorCheck:
    raw: pd.Series  # after warm-up trim
    corrected: pd.Series
    offset: float
    n_pairs: int


def calibration_pairs(cgm: pd.Series, fingerstick: pd.Series, start: pd.Timestamp,
                      end: pd.Timestamp) -> pd.DataFrame:
    """Fingerstick-sensor pairs with BOTH timestamps inside [start, end), so no reading
    after the calibration cutoff can be matched."""
    c = cgm[(cgm.index >= start) & (cgm.index < end)]
    f = fingerstick[(fingerstick.index >= start) & (fingerstick.index < end)]
    return pair_fingersticks(c, f)


def calibration_offset(cgm: pd.Series, fingerstick: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
                       prior: float) -> tuple[float, int]:
    """Person offset from calibration-interval pairs only, shrunk towards a training-only prior."""
    pairs = calibration_pairs(cgm, fingerstick, start, end)
    return person_offset(pairs, prior), len(pairs)


def check_sensor(cgm: pd.Series, fingerstick: pd.Series, prior: float) -> SensorCheck:
    raw = trim_warmup(cgm)
    pairs = pair_fingersticks(raw, fingerstick)
    offset = person_offset(pairs, prior)
    return SensorCheck(raw=raw, corrected=raw + offset, offset=offset, n_pairs=len(pairs))


def low_events(series: pd.Series, threshold: float = LOW_MG_DL, min_points: int = 2,
               max_gap_min: float = MAX_GAP_MIN) -> pd.DataFrame:
    """Runs of at least `min_points` consecutive readings below `threshold`."""
    cols = ["start", "end", "nadir", "points", "night"]
    if series.empty:
        return pd.DataFrame(columns=cols)
    s = series.dropna().sort_index()
    below = (s < threshold).to_numpy()
    times = s.index
    # Minutes since the previous reading, whatever the index's time resolution.
    gaps = np.r_[np.inf, np.diff(times.values).astype("timedelta64[s]").astype(float) / 60.0]
    rows, i, n = [], 0, len(s)
    while i < n:
        if not below[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and below[j + 1] and gaps[j + 1] <= max_gap_min:
            j += 1
        if j - i + 1 >= min_points:
            start = times[i]
            rows.append({"start": start, "end": times[j], "nadir": float(s.iloc[i:j + 1].min()),
                         "points": j - i + 1,
                         "night": NIGHT_START_HOUR <= start.hour < NIGHT_END_HOUR})
        i = j + 1
    return pd.DataFrame(rows, columns=cols)
