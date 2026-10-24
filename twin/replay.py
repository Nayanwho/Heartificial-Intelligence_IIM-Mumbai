"""Live replay: step a real, held-out ShanghaiT2DM sensor period through time.

At each replay clock the period is cut at the clock (`Period.until`), so later records cannot
reach any computation. The twin is calibrated on the fixed 5-day interval, features come from
the same code as the evaluation (twin/features.py), forecasts come from the saved population
model of the fold that EXCLUDED this patient, and alerts come from the shared alert policy
(twin/alerts.py) at that fold's training-selected threshold. Future readings are only loaded
separately, for display after the fact.

Which alert: in the version-2 evaluation (primary reference) a simple rule on the current
corrected reading caught at least as many lows as any model at the same alert budget, so the
replay alerts on that rule and shows the forecasts for context. The hybrid forecast rule is
available as `alert_score="hybrid"`.

What updates when: new CGM readings, doses and meals enter at every step (state assimilation).
The person's physiology core and sensor offset are fitted once, at the end of calibration, and
are not refitted online. A clinical flag (for example a medication start) changes the review
trigger, not the fitted physiology.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .alerts import AlertPolicy, emit
from .calibration import boundaries, calibrate
from .data_shanghai import Period
from .features import ARMS, eligible_origins, origin_features
from .inference import forecast, low_probability
from .protocol import GRID_MIN, HORIZONS_MIN

MODELS = Path(__file__).resolve().parents[1] / "results" / "models"
STALE_AFTER_MIN = 20


@dataclass
class FoldModels:
    fold: int
    forecast: dict  # horizon -> lightgbm.Booster
    classifier: object
    threshold: float  # alert threshold chosen on this fold's training patients (primary budget)
    features: list
    alert_score: str = "current"  # "current" (the corrected reading) or "hybrid" (30-minute forecast)

    @classmethod
    def load(cls, fold: int, threshold: float, models_dir: Path = MODELS, alert_score: str = "current") -> "FoldModels":
        import lightgbm as lgb
        fc = {h: lgb.Booster(model_file=str(models_dir / f"fold{fold}_C_{h}.txt")) for h in HORIZONS_MIN}
        clf = lgb.Booster(model_file=str(models_dir / f"fold{fold}_classifier.txt"))
        return cls(fold=fold, forecast=fc, classifier=clf, threshold=threshold, features=ARMS["C"],
                   alert_score=alert_score)


def predict_rows(models: FoldModels, feats: pd.DataFrame) -> pd.DataFrame:
    """Forecasts (mg/dL) at each horizon and the low probability, through the shared inference
    functions that the evaluation also uses (twin/inference.py)."""
    out = pd.DataFrame({"time": feats["time"].to_numpy(), "g0": feats["g0"].to_numpy()})
    for h, booster in models.forecast.items():
        out[f"C_{h}"] = forecast(booster, feats, models.features)
    out["p_low"] = low_probability(models.classifier, feats, models.features)
    return out


def replay_step(period: Period, prior: float, models: FoldModels, clock: pd.Timestamp,
                dropout_minutes: float = 0.0) -> dict:
    """Everything the dashboard shows at `clock`, computed only from records up to `clock`.

    `dropout_minutes` hides the last minutes of CGM before the clock, to show the
    insufficient-data path."""
    clock = pd.Timestamp(clock)
    warm_end, cal_end = boundaries(period)
    known = period.until(clock)
    if dropout_minutes > 0:
        known.cgm = known.cgm[known.cgm.index <= clock - pd.Timedelta(minutes=dropout_minutes)]
    base = {"clock": clock, "warm_end": warm_end, "cal_end": cal_end, "fold": models.fold,
            "threshold": models.threshold, "alert_score": models.alert_score}
    if clock < cal_end:
        done = max(0.0, (clock - warm_end).total_seconds() / 86400) if clock > warm_end else 0.0
        return {**base, "status": "calibrating",
                "message": f"Calibrating: {done:.1f} of 5 days collected. No forecast until calibration completes.",
                "history": known.cgm}
    cal = calibrate(known, prior)
    if cal.twin is None:
        return {**base, "status": "unavailable", "message": "The physiology core could not be fitted on calibration data.",
                "history": known.cgm}
    origins = eligible_origins(cal)
    origins = origins[cal.grid[origins] <= clock] if len(origins) else origins
    last_reading = known.cgm.index.max() if len(known.cgm) else None
    stale = last_reading is None or (clock - last_reading) > pd.Timedelta(minutes=STALE_AFTER_MIN)
    history = pd.Series(cal.ref, index=cal.grid).dropna()
    history = history[history.index <= clock]
    if not len(origins):
        return {**base, "status": "unavailable", "message": "Not enough CGM after calibration yet.",
                "history": history, "cal": cal}
    feats = origin_features(cal, origins)
    preds = predict_rows(models, feats)
    score = preds["g0"] if models.alert_score == "current" else preds["C_30"]
    fired = emit(preds["time"], score, preds["g0"], AlertPolicy(models.threshold, "glucose"))
    alerts = list(pd.DatetimeIndex(preds["time"][fired]))
    result = {**base, "cal": cal, "history": history, "alerts": alerts, "last_reading": last_reading,
              "inputs_logged": {"meals": int(len(known.meals)), "insulin_records": int(len(known.insulin)),
                                "tablet_records": int(len(known.tablets)),
                                "fingersticks": int(len(known.fingerstick))}}
    if stale or cal.grid[origins[-1]] < clock - pd.Timedelta(minutes=GRID_MIN):
        return {**result, "status": "unavailable",
                "message": f"Forecast unavailable: no CGM reading in the last {STALE_AFTER_MIN} minutes "
                           f"(last reading {last_reading:%d %b %H:%M})." if last_reading is not None else
                           "Forecast unavailable: no CGM reading yet."}
    now = preds.iloc[-1]
    origin = pd.Timestamp(now["time"])
    forecast = {h: float(now[f"C_{h}"]) for h in HORIZONS_MIN}
    return {**result, "status": "ok", "origin": origin, "g0": float(now["g0"]), "forecast": forecast,
            "p_low": float(now["p_low"]), "alert_now": bool(fired[-1]), "features": feats.iloc[-1].to_dict(),
            "message": "Forecast from the last reading, using records up to the replay clock."}


def clinical_snapshot(period: Period, clock: pd.Timestamp, synthetic_glp1_weeks: float | None = None) -> dict:
    """A trigger-ready record for a real patient, from the summary sheet and dose records up to the clock.
    An optional SYNTHETIC GLP-1 start (labelled as such) demonstrates a clinical-context update."""
    known = period.until(clock)
    info = period.info
    meds = []
    for drug in sorted(set(known.tablets["drug"])) if len(known.tablets) else []:
        meds.append((drug, 0.0, "as recorded", str(known.tablets.loc[known.tablets["drug"] == drug, "time"].min().date())))
    if len(known.insulin) or len(known.pump_bolus) or len(known.pump_basal):
        meds.append(("insulin", 0.0, "as recorded", str(period.cgm.index.min().date())))
    if synthetic_glp1_weeks is not None:
        start = pd.Timestamp(clock) - pd.Timedelta(weeks=synthetic_glp1_weeks)
        meds.append(("semaglutide", 0.5, "SYNTHETIC context update for the demo", str(start.date())))
    egfr = info.get("egfr")
    return {"id": period.period_id, "age": int(info.get("age")) if pd.notna(info.get("age")) else 0,
            "weight_kg_12w_ago": None, "weight_kg_now": None,
            "egfr_now": None if egfr is None or pd.isna(egfr) else float(egfr), "egfr_prev": None,
            "egfr_dates": [None, None], "cyp2c9": "not tested", "g6pd": "not tested",
            "medicines": meds, "fasting_plan": None}
