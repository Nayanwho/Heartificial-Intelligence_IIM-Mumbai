"""Personalise the twin from a fixed calibration interval, using nothing that comes later.

`calibrate(period, prior)` returns the state a deployed twin would hold at the end of the
5-day calibration interval: the person's frozen sensor offset, the meal fill value, the
baseline drug flags and clinical record, the input curves on a regular 15-minute grid, the
physiology core fitted on calibration data, and the person's typical day.

Everything in it is computed from records timestamped before the end of calibration, except
the input curves, whose value at any time uses only events logged at or before that time.
So `calibrate(period.until(t), prior)` and `calibrate(period, prior)` agree on everything a
computation made at time t can read (tested in tests/test_causal.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data_shanghai import Period
from .inputs import INPUT_COLUMNS, InputBank
from .meals import meal_carbs
from .model import Twin, fit_twin
from .protocol import (CALIBRATION_DAYS, FALLBACK_MEAL_CARBS, GRID_MIN, HORIZONS_MIN, MIN_CALIBRATION_MEALS,
                       MIN_CALIBRATION_READINGS, MIN_TEST_DAYS, WARMUP_HOURS)
from .sensor import calibration_offset, calibration_pairs

STATIC_FIELDS = ["age", "bmi", "hba1c", "egfr", "duration_years", "c_peptide", "male", "on_su", "on_insulin"]


@dataclass
class Calibration:
    period_id: str
    patient_id: str
    reference: str  # "corrected" or "raw"
    grid: pd.DatetimeIndex  # regular 15-minute grid from the first reading to the last reading + 120 min
    raw: np.ndarray  # raw CGM on the grid, NaN where missing or in warm-up
    ref: np.ndarray  # the reference series: raw + frozen offset (corrected) or raw
    warm_end: pd.Timestamp
    cal_end: pd.Timestamp
    prior: float
    offset: float
    n_pairs: int
    meal_fill: float
    n_cal_meals: int
    static: dict
    bank: InputBank
    inputs_now: np.ndarray  # [len(grid), 6] insulin, iob, su, glp1, carbs, cob, each from events up to that time
    twin: Twin | None
    typical: dict  # hour -> mean reference glucose during calibration
    cal_mean: float
    cal_sd: float
    last_cal_time: pd.Timestamp | None
    last_cal_value: float
    n_cal_readings: int
    meta: dict = field(default_factory=dict)

    def position(self, t: pd.Timestamp) -> int:
        return int(round((pd.Timestamp(t) - self.grid[0]) / pd.Timedelta(minutes=GRID_MIN)))

    @property
    def cal_end_pos(self) -> int:
        return self.position(self.cal_end)


def boundaries(period: Period) -> tuple[pd.Timestamp, pd.Timestamp]:
    first = period.cgm.index.min()
    warm_end = first + pd.Timedelta(hours=WARMUP_HOURS)
    return warm_end, warm_end + pd.Timedelta(days=CALIBRATION_DAYS)


def on_grid(series: pd.Series, grid: pd.DatetimeIndex) -> np.ndarray:
    """Place readings on the regular grid (nearest slot; readings are on-grid in ShanghaiT2DM)."""
    out = np.full(len(grid), np.nan)
    if series.empty:
        return out
    pos = np.round((series.index - grid[0]) / pd.Timedelta(minutes=GRID_MIN)).astype(int)
    keep = (pos >= 0) & (pos < len(grid))
    vals = pd.Series(series.to_numpy()[keep], index=pos[keep]).groupby(level=0).mean()
    out[vals.index.to_numpy()] = vals.to_numpy()
    return out


def static_record(period: Period, cal_end: pd.Timestamp) -> dict:
    info = period.info
    classes = period.drug_classes(before=cal_end)
    return {"age": info.get("age"), "bmi": info.get("bmi"), "hba1c": info.get("hba1c_percent"),
            "egfr": info.get("egfr"), "duration_years": info.get("duration_years"),
            "c_peptide": info.get("c_peptide_fasting"), "male": 1.0 if info.get("sex") == "M" else 0.0,
            "on_su": float("sulfonylurea" in classes), "on_insulin": float("insulin" in classes)}


def calibration_meal_fill(period: Period, start: pd.Timestamp, end: pd.Timestamp) -> tuple[float, int]:
    m = period.meals
    if m.empty:
        return FALLBACK_MEAL_CARBS, 0
    m = m[(m["time"] >= start) & (m["time"] < end) & m["logged"].astype(bool)]
    carbs = [c for c in (meal_carbs(items) for items in m["items"]) if c is not None]
    if len(carbs) >= MIN_CALIBRATION_MEALS:
        return float(np.median(carbs)), len(carbs)
    return FALLBACK_MEAL_CARBS, len(carbs)


def calibration_prior(periods, patients: set[str]) -> float:
    """Population offset prior: median fingerstick-minus-sensor gap over the calibration
    pairs of the given (training) patients only."""
    diffs = []
    for p in periods:
        if p.patient_id not in patients or p.cgm.empty:
            continue
        warm_end, cal_end = boundaries(p)
        pairs = calibration_pairs(p.cgm, p.fingerstick, warm_end, cal_end)
        diffs.extend(pairs["diff"].tolist())
    return float(np.median(diffs)) if diffs else 0.0


def calibrate(period: Period, prior: float, reference: str = "corrected") -> Calibration:
    first, last = period.cgm.index.min(), period.cgm.index.max()
    warm_end, cal_end = boundaries(period)
    grid = pd.date_range(first, last + pd.Timedelta(minutes=max(HORIZONS_MIN)), freq=f"{GRID_MIN}min")
    raw = on_grid(period.cgm, grid)
    raw[grid < warm_end] = np.nan
    if reference == "corrected":
        offset, n_pairs = calibration_offset(period.cgm, period.fingerstick, warm_end, cal_end, prior)
    elif reference == "raw":
        offset, n_pairs = 0.0, 0
    else:
        raise ValueError(reference)
    ref = raw + offset
    fill, n_meals = calibration_meal_fill(period, warm_end, cal_end)
    bank = InputBank(period, grid, fill_carbs=fill)
    full = bank.full()
    inputs_now = full[INPUT_COLUMNS].to_numpy()
    cal_mask = (grid >= warm_end) & (grid < cal_end)
    g_cal = pd.Series(ref[cal_mask], index=grid[cal_mask])
    n_cal = int(np.isfinite(g_cal.to_numpy()).sum())
    twin = None
    if n_cal >= 2:
        try:
            twin = fit_twin(g_cal, full.loc[cal_mask], meta={"period": period.period_id})
        except ValueError:
            twin = None
    valid = g_cal.dropna()
    typical = valid.groupby(valid.index.hour).mean().to_dict() if len(valid) else {}
    return Calibration(
        period_id=period.period_id, patient_id=period.patient_id, reference=reference, grid=grid, raw=raw,
        ref=ref, warm_end=warm_end, cal_end=cal_end, prior=float(prior), offset=float(offset), n_pairs=int(n_pairs),
        meal_fill=fill, n_cal_meals=n_meals, static=static_record(period, cal_end), bank=bank,
        inputs_now=inputs_now, twin=twin, typical=typical,
        cal_mean=float(valid.mean()) if len(valid) else np.nan, cal_sd=float(valid.std()) if len(valid) > 1 else np.nan,
        last_cal_time=valid.index[-1] if len(valid) else None,
        last_cal_value=float(valid.iloc[-1]) if len(valid) else np.nan, n_cal_readings=n_cal)


def eligibility(period: Period, cal: Calibration) -> tuple[bool, str]:
    """Dataset-selection rule for scoring (needs the period's length, so it is never a model input)."""
    last = period.cgm.index.max()
    if last < cal.cal_end + pd.Timedelta(days=MIN_TEST_DAYS):
        return False, f"less than {MIN_TEST_DAYS:g} day of CGM after the 5-day calibration"
    if cal.n_cal_readings < MIN_CALIBRATION_READINGS:
        return False, f"fewer than {MIN_CALIBRATION_READINGS} readings during calibration"
    if cal.twin is None:
        return False, "physiology core could not be fitted on calibration data"
    return True, "eligible"
