"""Illustrative dose scenarios on a person's physiology core (research feature).

A routine (meals and doses on a typical day) is repeated for a number of days and run
through the person's core under each dose option. Random process noise, scaled from the
core's own fitting error by an explicit `noise_factor`, turns each path into a spread of
simulated nights. The output is the share of simulated nights with a low (at least two
consecutive simulated values below 70 mg/dL between 00:00 and 06:00, the same 15-minute
definition used for real CGM episodes), plus simulated time in range.

These are COMPARATIVE, UNCALIBRATED numbers. They depend strongly on `noise_factor` and on
the body-change assumptions (appetite on a GLP-1 drug, extra drug effect after weight loss,
longer sulfonylurea action in kidney impairment, higher exposure with a CYP2C9 variant),
none of which has been fitted on patients. Causal treatment response has not been validated.
They are for clinician discussion, never a recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .data_shanghai import Period
from .drugs import DRUG_TO_CLASS
from .inputs import InputBank, Scenario
from .model import GLUCOSE_CEILING, GLUCOSE_FLOOR, Twin

NIGHT = (0, 6)
BURN_IN_DAYS = 2


@dataclass
class Routine:
    """A typical day. Hours are 0 to 24."""

    meals: list[tuple[float, float]]  # (hour, grams of carbohydrate)
    tablets: list[tuple[float, str, float]]  # (hour, drug, mg)
    insulin: list[tuple[float, dict, float]] = field(default_factory=list)  # (hour, components, units)
    weekly: list[tuple[int, float, str, float]] = field(default_factory=list)  # (weekday, hour, drug, mg)


def is_fasting_day(day: pd.Timestamp, fasting_hours, fasting_dates) -> bool:
    if fasting_hours is None:
        return False
    if not fasting_dates:
        return True  # a fast every simulated day
    return str(pd.Timestamp(day).date()) in set(fasting_dates)


def routine_period(routine: Routine, start: pd.Timestamp, days: int, fasting_hours=None,
                   fasting_dates: tuple[str, ...] = ()) -> Period:
    meal_rows, tablet_rows, insulin_rows = [], [], []
    for d in range(days):
        day = start + pd.Timedelta(days=d)
        fasting = is_fasting_day(day, fasting_hours, fasting_dates)
        for hour, carbs in routine.meals:
            if fasting and fasting_hours[0] <= hour < fasting_hours[1]:
                continue  # meals inside the fasting window are skipped on that date only
            meal_rows.append({"time": day + pd.Timedelta(hours=hour), "text": "routine meal",
                              "items": [], "logged": True, "carbs": float(carbs)})
        for hour, drug, mg in routine.tablets:
            tablet_rows.append({"time": day + pd.Timedelta(hours=hour), "drug": drug,
                                "drug_class": DRUG_TO_CLASS[drug], "dose_mg": float(mg)})
        for hour, components, units in routine.insulin:
            insulin_rows.append({"time": day + pd.Timedelta(hours=hour), "product": "routine",
                                 "units": float(units), "route": "sc", "components": components})
        for weekday, hour, drug, mg in routine.weekly:
            if day.weekday() == weekday:
                tablet_rows.append({"time": day + pd.Timedelta(hours=hour), "drug": drug,
                                    "drug_class": DRUG_TO_CLASS[drug], "dose_mg": float(mg)})
    grid = pd.date_range(start, periods=days * 96, freq="15min")
    return Period(period_id="routine", patient_id="routine", cgm=pd.Series(0.0, index=grid),
                  fingerstick=pd.Series(dtype=float),
                  meals=pd.DataFrame(meal_rows, columns=["time", "text", "items", "logged", "carbs"]),
                  insulin=pd.DataFrame(insulin_rows, columns=["time", "product", "units", "route", "components"]),
                  pump_bolus=pd.Series(dtype=float), pump_basal=pd.Series(dtype=float),
                  tablets=pd.DataFrame(tablet_rows, columns=["time", "drug", "drug_class", "dose_mg"]), info={})


@dataclass
class BodyChange:
    """Assumed effects of a trigger, applied on top of a dose scenario. All are assumptions."""

    appetite_cut: float = 0.0  # share of meal carbohydrate no longer eaten (GLP-1 start)
    sensitivity_gain: float = 0.0  # extra insulin and sulfonylurea effect (weight loss)
    su_half_life: float = 1.0  # longer sulfonylurea action (stand-in for active-metabolite build-up)
    insulin_duration: float = 1.0  # longer insulin action (kidney decline)
    su_exposure: float = 1.0  # higher exposure per mg (CYP2C9 variant)
    fasting_hours: tuple[int, int] | None = None  # no meals in this window on fasting days
    fasting_dates: tuple[str, ...] = ()  # ISO dates of planned fasts; empty with hours set = every day


def adjusted_twin(twin: Twin, change: BodyChange) -> Twin:
    coef = dict(twin.coef)
    coef["insulin"] *= 1 + change.sensitivity_gain
    coef["su"] *= (1 + change.sensitivity_gain) * change.su_exposure
    return replace(twin, coef=coef)


def simulate_paths(twin: Twin, inputs: np.ndarray, start: float, n_sims: int, noise_sd: float,
                   seed: int = 0) -> np.ndarray:
    """[n_sims + 1, T] glucose paths; row 0 is the noise-free path."""
    c = twin.coef
    w = np.array([c["insulin"], c["su"], c["glp1"], c["carbs"], c["sin24"], c["cos24"]])
    drive = inputs @ w + c["const"]
    rng = np.random.default_rng(seed)
    paths = np.empty((n_sims + 1, len(inputs)))
    g = np.full(n_sims + 1, float(start))
    paths[:, 0] = g
    for i in range(1, len(inputs)):
        noise = rng.normal(0.0, noise_sd, n_sims + 1)
        noise[0] = 0.0
        g = np.clip(g + drive[i - 1] + c["glucose"] * g + noise, GLUCOSE_FLOOR, GLUCOSE_CEILING)
        paths[:, i] = g
    return paths


DEMO_AS_OF = pd.Timestamp("2026-10-20")  # the demo's "today"; the dashboard shows and can change it
NOISE_FACTORS = (0.25, 0.5, 1.0)  # sensitivity grid; 0.5 is the default display setting, not a fitted value


def night_has_low(paths: np.ndarray, cols: np.ndarray, threshold: float = 70.0) -> np.ndarray:
    """Per simulated path: does the night contain at least two consecutive values below 70?"""
    seg = paths[:, cols] < threshold
    if seg.shape[1] < 2:
        return np.zeros(len(paths), dtype=bool)
    return (seg[:, 1:] & seg[:, :-1]).any(axis=1)


def rehearse(twin: Twin, routine: Routine, scenario: Scenario, change: BodyChange | None = None,
             days: int = 14, n_sims: int = 300, noise_factor: float = 0.5, seed: int = 0,
             start: str | pd.Timestamp | None = None) -> dict:
    """Run a routine for `days` from `start` (default: the day after DEMO_AS_OF) under a dose
    scenario and body change; summarise simulated lows. Comparative and uncalibrated."""
    change = change or BodyChange()
    start = pd.Timestamp(start) if start is not None else DEMO_AS_OF + pd.Timedelta(days=1)
    t0 = start - pd.Timedelta(days=BURN_IN_DAYS)
    total_days = days + BURN_IN_DAYS
    period = routine_period(routine, t0, total_days, change.fasting_hours, change.fasting_dates)
    times = period.cgm.index
    sc = replace(scenario, su_half_life=scenario.su_half_life * change.su_half_life,
                 insulin_duration=scenario.insulin_duration * change.insulin_duration,
                 meal_size=scenario.meal_size * (1 - change.appetite_cut))
    bank = InputBank(period, times, scenario=sc)
    inputs = bank.full()[["insulin", "su", "glp1", "carbs", "sin24", "cos24"]].to_numpy()
    tw = adjusted_twin(twin, change)
    start_glucose = min(max(tw.basal_glucose, 90.0), 250.0)
    paths = simulate_paths(tw, inputs, start_glucose, n_sims, noise_factor * twin.residual_sd, seed)
    keep = times >= start
    paths, times = paths[:, keep], times[keep]
    hours = times.hour
    night = (hours >= NIGHT[0]) & (hours < NIGHT[1])
    nights = pd.Series(times.normalize()[night]).unique()
    night_risk = []
    for n in nights:
        cols = np.flatnonzero(night & (times.normalize() == n))
        night_risk.append(float(night_has_low(paths[1:], cols).mean()))
    sims = paths[1:]
    return {
        "times": times, "path": paths[0], "p10": np.percentile(sims, 10, axis=0),
        "p90": np.percentile(sims, 90, axis=0),
        "night_low_risk": float(np.mean(night_risk)) if night_risk else 0.0,
        "night_low_risk_by_night": night_risk, "nights": len(night_risk), "noise_factor": noise_factor,
        "nights_with_low_expected": float(np.sum(night_risk)),
        "any_low_risk_per_day": float(np.mean([night_has_low(sims, np.flatnonzero(times.normalize() == d)).mean()
                                               for d in pd.Series(times.normalize()).unique()])),
        "mean_glucose": float(sims.mean()),
        "time_in_range_percent": float(((sims >= 70) & (sims <= 180)).mean() * 100),
        "time_above_180_percent": float((sims > 180).mean() * 100),
        "time_below_70_percent": float((sims < 70).mean() * 100),
    }


def compare(twin: Twin, options: dict[str, Routine], change: BodyChange | None = None,
            scenario: Scenario | None = None, **kwargs) -> pd.DataFrame:
    """One row per dose option (each a routine) under the same body change."""
    rows = []
    for name, routine in options.items():
        r = rehearse(twin, routine, scenario or Scenario(), change, **kwargs)
        rows.append({"option": name, "night_low_risk_percent": round(100 * r["night_low_risk"], 1),
                     "nights_simulated": r["nights"], "noise_factor": r["noise_factor"],
                     "nights_with_a_low_in_14": round(r["nights_with_low_expected"], 1),
                     "mean_glucose_mg_dl": round(r["mean_glucose"]),
                     "time_in_range_percent": round(r["time_in_range_percent"]),
                     "time_above_180_percent": round(r["time_above_180_percent"]),
                     "time_below_70_percent": round(r["time_below_70_percent"], 1)})
    return pd.DataFrame(rows)

