"""Turn logged doses and meals into the time courses the twin reads.

At every sensor time the twin needs to know how much insulin is acting, how much
sulfonylurea and GLP-1 drug is circulating, and how much meal carbohydrate is
entering the blood. Each logged event is spread over time with a typical action
curve; the twin's fitted coefficients then scale these curves to the person.

`InputBank` keeps each event's contribution separately, so a forecast made at
time t can use only the doses and meals logged up to t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import gamma

from .data_shanghai import Period
from .drugs import INSULIN_PROFILES, SULFONYLUREAS
from .meals import meal_carbs

# Action-curve shapes: gamma curves peaking at the profile's typical peak time.
GAMMA_SHAPE = {"rapid": 3, "regular": 6, "nph": 8, "iv": 2}
# Basal analogues act almost flat: (ramp hours, total duration hours).
FLAT_PROFILES = {"long": (1.0, 24.0), "ultralong": (1.0, 42.0)}
MEAL_PEAK_H, MEAL_SHAPE = 0.75, 3

# Sulfonylurea reference doses (mg) that count as one exposure unit.
SU_REFERENCE_MG = {"glimepiride": 2.0, "gliclazide": 60.0, "glipizide": 5.0,
                   "gliquidone": 30.0, "glibenclamide": 5.0, "glyburide": 5.0}
SU_ABSORPTION_PER_H = 1.0
GLP1_HALF_LIFE_H = {"liraglutide": 13.0, "exenatide": 2.4, "semaglutide": 168.0, "dulaglutide": 120.0,
                    "loxenatide": 120.0, "beinaglutide": 0.2, "lixisenatide": 3.0, "tirzepatide": 120.0}
GLP1_REFERENCE_MG = {"liraglutide": 1.2, "semaglutide": 0.5, "dulaglutide": 1.5, "exenatide": 0.01,
                     "loxenatide": 0.2, "beinaglutide": 0.1, "lixisenatide": 0.02, "tirzepatide": 5.0}
GLP1_ABSORPTION_PER_H = 0.08
INPUT_COLUMNS = ["insulin", "iob", "su", "glp1", "carbs", "cob"]


def insulin_kernel(profile: str, tau_h: np.ndarray) -> np.ndarray:
    """Insulin acting per hour, per unit injected, tau_h hours after the dose."""
    tau = np.asarray(tau_h, dtype=float)
    if profile in FLAT_PROFILES:
        ramp, duration = FLAT_PROFILES[profile]
        rise = np.clip(tau / ramp, 0, 1)
        fall = np.clip((duration - tau) / ramp, 0, 1)
        shape = np.where((tau > 0) & (tau < duration), np.minimum(rise, fall), 0.0)
        return shape / (duration - ramp)  # the trapezoid's area is duration - ramp
    k = GAMMA_SHAPE[profile]
    theta = INSULIN_PROFILES[profile]["peak_h"] / (k - 1)
    return np.where(tau > 0, gamma.pdf(np.clip(tau, 0, None), k, scale=theta), 0.0)


def insulin_remaining(profile: str, tau_h: np.ndarray) -> np.ndarray:
    """Fraction of a delivered dose still to act (insulin on board); zero before the dose."""
    tau = np.asarray(tau_h, dtype=float)
    if profile in FLAT_PROFILES:
        _, duration = FLAT_PROFILES[profile]
        return np.where(tau <= 0, 0.0, np.clip(1 - tau / duration, 0, 1))
    k = GAMMA_SHAPE[profile]
    theta = INSULIN_PROFILES[profile]["peak_h"] / (k - 1)
    return np.where(tau <= 0, 0.0, gamma.sf(np.clip(tau, 0, None), k, scale=theta))


def meal_kernel(tau_h: np.ndarray) -> np.ndarray:
    """Carbohydrate entering the blood per hour, per gram eaten."""
    tau = np.asarray(tau_h, dtype=float)
    theta = MEAL_PEAK_H / (MEAL_SHAPE - 1)
    return np.where(tau > 0, gamma.pdf(np.clip(tau, 0, None), MEAL_SHAPE, scale=theta), 0.0)


def meal_remaining(tau_h: np.ndarray) -> np.ndarray:
    tau = np.asarray(tau_h, dtype=float)
    theta = MEAL_PEAK_H / (MEAL_SHAPE - 1)
    return np.where(tau <= 0, 0.0, gamma.sf(np.clip(tau, 0, None), MEAL_SHAPE, scale=theta))


def bateman(tau_h: np.ndarray, half_life_h: float, ka: float) -> np.ndarray:
    """One-compartment exposure curve after an oral or injected dose."""
    tau = np.clip(np.asarray(tau_h, dtype=float), 0, None)
    ke = np.log(2) / half_life_h
    if abs(ka - ke) < 1e-9:
        ka += 1e-6
    curve = ka / (ka - ke) * (np.exp(-ke * tau) - np.exp(-ka * tau))
    return np.where(np.asarray(tau_h) > 0, curve, 0.0)


def hours_between(times: pd.DatetimeIndex, events) -> np.ndarray:
    t = times.values.astype("datetime64[s]").astype(np.int64)[:, None]
    e = pd.DatetimeIndex(events).values.astype("datetime64[s]").astype(np.int64)[None, :]
    return (t - e) / 3600.0


def meal_table(period: Period, fallback_carbs: float, fill_carbs: float | None = None) -> pd.DataFrame:
    """Meals with estimated carbs; unweighed or unlogged meals get a fill value.

    With `fill_carbs` (the causal evaluation passes the median of meals logged during
    calibration), that value is used. Without it, the person's median over the records given
    is used, which looks at later meals; the version-2 evaluation and replay always pass it."""
    if period.meals.empty:
        return pd.DataFrame(columns=["time", "carbs", "imputed"])
    m = period.meals.copy()
    if "carbs" in m.columns and m["carbs"].notna().all():  # simulated routines give carbs directly
        m["imputed"] = False
        return m[["time", "carbs", "imputed"]].reset_index(drop=True)
    m["carbs"] = [meal_carbs(items) if logged else None for items, logged in zip(m["items"], m["logged"])]
    known = m["carbs"].dropna()
    if fill_carbs is not None:
        fill = float(fill_carbs)
    else:
        fill = float(known.median()) if len(known) >= 3 else fallback_carbs
    m["imputed"] = m["carbs"].isna()
    m["carbs"] = m["carbs"].fillna(fill).astype(float)
    return m[["time", "carbs", "imputed"]].reset_index(drop=True)


def insulin_events(period: Period, cap_basal_h: float = 6.0) -> pd.DataFrame:
    """Every insulin delivery as (time, profile, units), pump basal split into 15-minute pulses.

    A basal rate runs from its record until the next record, and for at most `cap_basal_h` hours
    after its own record. The pulse at any time depends only on records at or before that time,
    so a record cut at a replay clock gives the same pulses up to the clock as the full record."""
    rows = []
    for time, units, components in period.insulin[["time", "units", "components"]].itertuples(index=False):
        for profile, frac in components.items():
            rows.append((time, profile, units * frac))
    for time, units in period.pump_bolus.items():
        if units > 0:
            rows.append((time, "regular", float(units)))
    basal = period.pump_basal
    if len(basal):
        cap = pd.Timedelta(hours=cap_basal_h)
        nexts = list(basal.index[1:]) + [basal.index[-1] + cap]
        stops = [min(nxt, start + cap) for start, nxt in zip(basal.index, nexts)]
        for (start, rate), stop in zip(basal.items(), stops):
            if rate <= 0 or stop <= start:
                continue
            for pulse in pd.date_range(start, stop, freq="15min", inclusive="left"):
                rows.append((pulse, "regular", float(rate) * 0.25))
    return pd.DataFrame(rows, columns=["time", "profile", "units"])


@dataclass
class Scenario:
    """Changes applied in a what-if. 1.0 means unchanged."""

    su_dose: float = 1.0  # multiplies every sulfonylurea dose
    insulin_dose: float = 1.0  # multiplies every insulin dose
    meal_size: float = 1.0  # multiplies every meal's carbohydrate
    su_half_life: float = 1.0  # slower clearance (kidney decline, CYP2C9 poor metaboliser)
    insulin_duration: float = 1.0  # longer insulin action (kidney decline)
    skip_meals_between: tuple[int, int] | None = None  # (start hour, end hour) with no meals: fasting


class InputBank:
    """Per-event contributions to each input at every sensor time."""

    def __init__(self, period: Period, times: pd.DatetimeIndex, fallback_carbs: float = 45.0,
                 scenario: Scenario | None = None, fill_carbs: float | None = None):
        s = scenario or Scenario()
        self.times = times
        parts: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {c: [] for c in INPUT_COLUMNS}

        ins = insulin_events(period)
        for profile, group in ins.groupby("profile"):
            tau = hours_between(times, group["time"]) / s.insulin_duration
            units = group["units"].to_numpy()[None, :] * s.insulin_dose
            et = group["time"].to_numpy()
            parts["insulin"].append((et, insulin_kernel(profile, tau) / s.insulin_duration * units))
            parts["iob"].append((et, insulin_remaining(profile, tau) * units))

        su = period.tablets[period.tablets["drug_class"] == "sulfonylurea"].dropna(subset=["dose_mg"])
        for drug, group in su.groupby("drug"):
            half_life = float(SULFONYLUREAS[drug]["half_life_h"]) * s.su_half_life
            doses = (group["dose_mg"] / SU_REFERENCE_MG[drug]).to_numpy()[None, :] * s.su_dose
            parts["su"].append((group["time"].to_numpy(), bateman(hours_between(times, group["time"]),
                                                                  half_life, SU_ABSORPTION_PER_H) * doses))

        glp = period.tablets[period.tablets["drug_class"] == "glp1"].dropna(subset=["dose_mg"])
        for drug, group in glp.groupby("drug"):
            doses = (group["dose_mg"] / GLP1_REFERENCE_MG.get(drug, 1.0)).to_numpy()[None, :]
            parts["glp1"].append((group["time"].to_numpy(), bateman(hours_between(times, group["time"]),
                                                                    GLP1_HALF_LIFE_H.get(drug, 13.0),
                                                                    GLP1_ABSORPTION_PER_H) * doses))

        meals = meal_table(period, fallback_carbs, fill_carbs)
        if len(meals) and s.skip_meals_between is not None:
            lo, hi = s.skip_meals_between
            hour = pd.DatetimeIndex(meals["time"]).hour
            meals = meals[~((hour >= lo) & (hour < hi))]
        self.meals = meals
        if len(meals):
            tau = hours_between(times, meals["time"])
            grams = meals["carbs"].to_numpy()[None, :] * s.meal_size
            et = meals["time"].to_numpy()
            parts["carbs"].append((et, meal_kernel(tau) * grams))
            parts["cob"].append((et, meal_remaining(tau) * grams))

        # Stack per input, sort events by time, keep cumulative sums for causal look-ups.
        self.event_times: dict[str, np.ndarray] = {}
        self.cumulative: dict[str, np.ndarray] = {}
        for col, plist in parts.items():
            if not plist:
                self.event_times[col] = np.array([], dtype="datetime64[ns]")
                self.cumulative[col] = np.zeros((len(times), 1))
                continue
            et = np.concatenate([p[0] for p in plist]).astype("datetime64[ns]")
            mat = np.concatenate([p[1] for p in plist], axis=1)
            order = np.argsort(et, kind="stable")
            self.event_times[col] = et[order]
            self.cumulative[col] = np.concatenate([np.zeros((len(times), 1)),
                                                   np.cumsum(mat[:, order], axis=1)], axis=1)
        hours = times.hour + times.minute / 60.0
        self.sin24 = np.sin(2 * np.pi * hours / 24.0)
        self.cos24 = np.cos(2 * np.pi * hours / 24.0)

    def full(self) -> pd.DataFrame:
        """Inputs using every logged event (for fitting and for what-if simulation)."""
        out = pd.DataFrame({c: self.cumulative[c][:, -1] for c in INPUT_COLUMNS}, index=self.times)
        out["sin24"], out["cos24"] = self.sin24, self.cos24
        return out

    def known_array(self, origin_pos: int, idx: np.ndarray) -> np.ndarray:
        """Like `known_at`, as an array [rows x 6] (insulin, su, glp1, carbs, sin24, cos24), for speed.

        `origin_pos` is the row of the forecast origin; events logged up to that time count.
        """
        origin64 = self.times.values[origin_pos]
        cols = []
        for c in ("insulin", "su", "glp1", "carbs"):
            k = int(np.searchsorted(self.event_times[c], origin64, side="right"))
            cols.append(self.cumulative[c][idx, k])
        return np.column_stack(cols + [self.sin24[idx], self.cos24[idx]])

    def known_at(self, origin: pd.Timestamp, rows: slice | np.ndarray) -> pd.DataFrame:
        """Inputs at `rows` using only events logged at or before `origin`."""
        idx = np.arange(len(self.times))[rows]
        origin64 = np.datetime64(pd.Timestamp(origin).as_unit("ns"))
        data = {}
        for c in INPUT_COLUMNS:
            k = int(np.searchsorted(self.event_times[c], origin64, side="right"))
            data[c] = self.cumulative[c][idx, k]
        out = pd.DataFrame(data, index=self.times[idx])
        out["sin24"], out["cos24"] = self.sin24[idx], self.cos24[idx]
        return out
