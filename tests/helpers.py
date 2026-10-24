"""A small synthetic sensor period for fast, deterministic tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twin.data_shanghai import Period  # noqa: E402


def synthetic_period(days: float = 9.0, seed: int = 0, period_id: str = "9001_0_20210101") -> Period:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2021-01-01 00:00", periods=int(days * 96), freq="15min")
    hours = (times.hour + times.minute / 60).to_numpy()
    glucose = 140 + 35 * np.sin(2 * np.pi * (hours - 8) / 24) + rng.normal(0, 8, len(times))
    for d in range(int(days)):  # a few lows at night
        glucose[d * 96 + 12:d * 96 + 15] = 62.0
    cgm = pd.Series(glucose, index=times, name="cgm")
    fs_times = [t for t in times if t.hour in (7, 11, 17, 21) and t.minute == 0]
    fingerstick = pd.Series(cgm.loc[fs_times].to_numpy() + 9.0, index=pd.DatetimeIndex(fs_times))
    meal_times = [t for t in times if (t.hour, t.minute) in ((8, 0), (12, 30), (19, 0))]
    meals = pd.DataFrame({"time": meal_times, "text": "rice 150g",
                          "items": [[("rice", 150.0, "g")]] * len(meal_times), "logged": True})
    tab_times = [t for t in times if (t.hour, t.minute) in ((7, 45), (18, 45))]
    tablets = pd.DataFrame({"time": tab_times, "drug": "glimepiride", "drug_class": "sulfonylurea", "dose_mg": 2.0})
    insulin = pd.DataFrame({"time": [t for t in times if (t.hour, t.minute) == (21, 0)], "product": "glargine",
                            "units": 10.0, "route": "sc", "components": [{"long": 1.0}] *
                            len([t for t in times if (t.hour, t.minute) == (21, 0)])})
    info = {"age": 62, "bmi": 26.0, "hba1c_percent": 8.1, "egfr": 70, "duration_years": 9,
            "c_peptide_fasting": 0.6, "sex": "F", "agents": ""}
    return Period(period_id=period_id, patient_id=period_id[:4], cgm=cgm, fingerstick=fingerstick, meals=meals,
                  insulin=insulin, pump_bolus=pd.Series(dtype=float), pump_basal=pd.Series(dtype=float),
                  tablets=tablets, info=info)
