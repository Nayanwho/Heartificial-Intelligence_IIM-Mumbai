"""Experimental sensor-off glucose estimation.

After the 5-day calibration the CGM is hidden. The physiology core starts from the LAST
calibration reading and sees only logged doses and meals, re-anchoring at fingersticks.
CGM readings after calibration are used only as the scoring reference, never as a state or
a feature. Acute low alerts are not available in this mode: in testing it estimated almost no
real lows below 80 mg/dL. Coverage in the open data is at most about 8 days per period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import Calibration
from .data_shanghai import Period
from .protocol import SENSOR_OFF_MIN_DAYS, SENSOR_OFF_MIN_FINGERSTICKS

FEATURES = ["twin", "typical", "carried", "hours_since_anchor", "cal_mean", "cal_sd", "iob", "cob", "su_now",
            "insulin_now", "carbs_now", "hour_sin", "hour_cos", "twin_minus_typical"]


def sensor_off_frame(cal: Calibration, period: Period) -> pd.DataFrame | None:
    """Estimates for every post-calibration grid slot, plus the hidden reference for scoring."""
    if cal.twin is None or cal.last_cal_time is None:
        return None
    last_reading = period.cgm.index.max()
    fs = period.fingerstick[(period.fingerstick.index >= cal.cal_end) & (period.fingerstick.index <= last_reading)]
    fs = fs[~fs.index.duplicated(keep="last")]
    if len(fs) < SENSOR_OFF_MIN_FINGERSTICKS or (last_reading - cal.cal_end) < pd.Timedelta(days=SENSOR_OFF_MIN_DAYS):
        return None
    p0 = cal.position(cal.last_cal_time)
    p_end = cal.position(last_reading)
    full = cal.bank.full()
    inputs = full.iloc[p0:p_end + 1]
    path = cal.twin.simulate(cal.last_cal_value, inputs, anchors=fs)  # never reads CGM after p0
    times = inputs.index
    anchors = pd.concat([pd.Series([cal.last_cal_value], index=[cal.last_cal_time]), fs]).sort_index()
    anchor_idx = anchors.index.searchsorted(times, side="right") - 1
    carried = anchors.to_numpy()[anchor_idx]
    hours_since = (times - anchors.index[anchor_idx]).total_seconds().to_numpy() / 3600.0
    typical = np.array([cal.typical.get(h, cal.cal_mean) for h in times.hour])
    hour = times.hour + times.minute / 60.0
    frame = pd.DataFrame({
        "period": cal.period_id, "patient": cal.patient_id, "time": times, "twin": path.to_numpy(),
        "typical": typical, "carried": carried, "hours_since_anchor": hours_since, "cal_mean": cal.cal_mean,
        "cal_sd": cal.cal_sd, "iob": inputs["iob"].to_numpy(), "cob": inputs["cob"].to_numpy(),
        "su_now": inputs["su"].to_numpy(), "insulin_now": inputs["insulin"].to_numpy(),
        "carbs_now": inputs["carbs"].to_numpy(), "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24)})
    frame["twin_minus_typical"] = frame["twin"] - frame["typical"]
    for k, v in cal.static.items():
        frame[k] = v
    # Hidden reference, for scoring only. Score only after the calibration cutoff.
    frame["truth"] = cal.ref[p0:p_end + 1]
    frame = frame[(frame["time"] >= cal.cal_end) & frame["truth"].notna()].reset_index(drop=True)
    return frame
