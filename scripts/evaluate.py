"""Version-2 evaluation on real patients (docs/evaluation_protocol.md).

Forecasts, a fair alarm comparison at equal alert budgets, the clinical-record ablation,
and experimental sensor-off estimation. Every population quantity is fitted on outer-training
patients only; every person is personalised from their own 5-day calibration interval only.

Run from the repository root:  python scripts/evaluate.py      (about 10 to 20 minutes)
Outputs in results/: evaluation.json, evaluation_raw.json, folds.json, calibrations.csv,
exclusions.csv, predictions.csv.gz, alerts.csv.gz, events.csv, sensor_off.csv.gz, twins.csv,
models/, manifest.json.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from twin.alerts import AlertPolicy, emit, score_alerts  # noqa: E402
from twin.calibration import calibrate, calibration_prior, eligibility  # noqa: E402
from twin.data_shanghai import load_all  # noqa: E402
from twin.features import ARMS, eligible_origins, origin_features, targets  # noqa: E402
from twin.inference import booster_of, forecast, low_probability  # noqa: E402
from twin.metrics import clarke_zones  # noqa: E402
from twin.protocol import (ALERT_BUDGETS_PER_DAY, ALERT_COOLDOWN_MIN, BOOTSTRAP_REPS, GRID_MIN,  # noqa: E402
                           HORIZONS_MIN, LGB_PARAMS, LOW_MG_DL, N_INNER_FOLDS, N_OUTER_FOLDS, PRIMARY_BUDGET,
                           PROTOCOL_VERSION, SEED, WARNING_WINDOW_MIN)
from twin.sensor import low_events  # noqa: E402
from twin.sensor_off import FEATURES as OFF_FEATURES, sensor_off_frame  # noqa: E402
from twin.features import EHR  # noqa: E402

OUT = ROOT / "results"
MODELS = OUT / "models"
FORECAST_METHODS = ["persistence", "trend", "physiology", "A", "B", "D", "C"]
METHOD_NAMES = {"persistence": "No change (persistence)", "trend": "Recent trend extrapolated",
                "physiology": "Physiology core only", "A": "ML: dynamic only",
                "B": "ML: dynamic + clinical record", "D": "ML: dynamic + physiology",
                "C": "Hybrid: dynamic + clinical record + physiology"}
ALERT_METHODS = {"current": ("g0", "glucose"), "trend": ("trend_30", "glucose"),
                 "physiology": ("phys_30", "glucose"), "ml": ("B_30", "glucose"),
                 "hybrid": ("C_30", "glucose"), "classifier": ("p_low", "probability")}
ALERT_NAMES = {"current": "Current reading (= persistence forecast)", "trend": "Trend forecast, 30 min",
               "physiology": "Physiology core forecast, 30 min", "ml": "ML only (dynamic + record), 30 min",
               "hybrid": "Hybrid forecast, 30 min", "classifier": "Hybrid classifier, low within 60 min"}
GLUCOSE_CANDIDATES = np.arange(40.0, 160.5, 0.5)


# ---------------------------------------------------------------- folds and models

def assign_folds(patients: list[str]) -> dict[str, int]:
    rng = np.random.default_rng(SEED)
    order = rng.permutation(sorted(patients))
    return {str(p): int(i % N_OUTER_FOLDS) for i, p in enumerate(order)}


def fit_lgb(X: pd.DataFrame, y: pd.Series, binary: bool = False):
    model = lgb.LGBMClassifier(**LGB_PARAMS) if binary else lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X, y)
    return model


def inner_plan(patients: list[str]) -> list[tuple[set, set]]:
    """Inner patient folds inside an outer-training set: (fitting patients, held-out patients) per fold."""
    order = np.random.default_rng(SEED + 1).permutation(sorted(patients))
    inner = {str(p): i % N_INNER_FOLDS for i, p in enumerate(order)}
    return [({p for p, f in inner.items() if f != j}, {p for p, f in inner.items() if f == j})
            for j in range(N_INNER_FOLDS)]


def inner_learned_scores(periods, train: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Out-of-fold learned scores (B_30, C_30, p_low) for the outer-training patients.

    Each inner fold has the same isolation as the outer test (protocol section 5): the offset
    prior is recomputed from that inner fold's fitting patients only, every outer-training period
    is recalibrated with it, the models are fitted on the fitting patients, and the held-out inner
    patients are scored. With the raw reference the prior is zero, so the outer rows are reused."""
    pats = sorted(train["patient"].unique())
    by_patient = {}
    for p in periods:
        if p.patient_id in pats and not p.cgm.empty:
            by_patient.setdefault(p.patient_id, []).append(p)
    frames = []
    for fit_pats, val_pats in inner_plan(pats):
        if reference == "corrected":
            prior = calibration_prior(periods, fit_pats)
            rows = []
            for pid in pats:
                for p in by_patient.get(pid, []):
                    cal = calibrate(p, prior, reference)
                    ok, _ = eligibility(p, cal)
                    if ok:
                        rows.append(period_rows(cal, p)[0])
            rows = pd.concat(rows, ignore_index=True)
        else:
            rows = train.copy()
        fit = rows[rows["patient"].isin(fit_pats)]
        val = rows[rows["patient"].isin(val_pats)].copy()
        for arm in ("B", "C"):
            ok = fit["y_30"].notna()
            val[f"{arm}_30"] = forecast(fit_lgb(fit.loc[ok, ARMS[arm]], fit.loc[ok, "y_30"]), val, ARMS[arm])
        val["p_low"] = low_probability(fit_lgb(fit[ARMS["C"]], fit["low_next60"], binary=True), val, ARMS["C"])
        frames.append(val[["period", "patient", "time", "g0", "B_30", "C_30", "p_low"]])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- rows per period

def period_rows(cal, period) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos = eligible_origins(cal)
    feats = origin_features(cal, pos)
    tg = targets(cal, pos)
    rows = pd.concat([feats, tg], axis=1)
    series = pd.Series(cal.ref, index=cal.grid).dropna()
    ev = low_events(series, LOW_MG_DL)
    ev = ev[ev["start"] >= cal.cal_end] if len(ev) else ev
    events = pd.DataFrame({"period": cal.period_id, "patient": cal.patient_id,
                           "event_start": ev["start"] if len(ev) else pd.Series(dtype="datetime64[ns]"),
                           "night": ev["night"] if len(ev) else pd.Series(dtype=bool)})
    events["eligible"] = events["event_start"] >= cal.cal_end + pd.Timedelta(minutes=WARNING_WINDOW_MIN)
    starts = pd.DatetimeIndex(events["event_start"])
    t = pd.DatetimeIndex(rows["time"]) if len(rows) else pd.DatetimeIndex([])
    lab = np.zeros(len(rows), dtype=int)
    for s in starts:
        lab |= ((s - t > pd.Timedelta(0)) & (s - t <= pd.Timedelta(minutes=WARNING_WINDOW_MIN))).astype(int)
    rows["low_next60"] = lab
    for h in HORIZONS_MIN:
        rows[f"phys_{h}"] = rows["g0"] + rows[f"twin_{h}"]
    return rows, events


# ---------------------------------------------------------------- alert thresholds

class RateScanner:
    """Emitted alerts per monitored day for many thresholds, without outcomes."""

    def __init__(self, frame: pd.DataFrame, score_col: str):
        self.groups = []
        for _, g in frame.groupby("period", sort=False):
            g = g.sort_values("time")
            minutes = (pd.DatetimeIndex(g["time"]) - g["time"].iloc[0]).total_seconds().to_numpy() / 60.0
            self.groups.append((minutes, g[score_col].to_numpy(float), g["g0"].to_numpy(float)))
        self.days = len(frame) * GRID_MIN / 1440.0

    def rate(self, policy: AlertPolicy) -> float:
        n = 0
        for minutes, score, cur in self.groups:
            last = -np.inf
            for k in np.flatnonzero(policy.condition(score, cur)):
                if minutes[k] - last >= policy.cooldown_min:
                    n += 1
                    last = minutes[k]
        return n / self.days


def select_threshold(frame: pd.DataFrame, score_col: str, kind: str, budget: float) -> float:
    scanner = RateScanner(frame, score_col)
    if kind == "glucose":
        candidates = GLUCOSE_CANDIDATES  # strict (low) to permissive (high)
    else:
        q = np.unique(np.quantile(frame[score_col].dropna(), np.linspace(0.5, 0.99999, 2000)))
        candidates = q[::-1]  # strict (high probability) to permissive (low)
    best = float(candidates[0])
    for thr in candidates:
        if scanner.rate(AlertPolicy(float(thr), kind)) <= budget:
            best = float(thr)
        else:
            break
    return best


# ---------------------------------------------------------------- bootstrap helpers

def weights(n_patients: int, reps: int = BOOTSTRAP_REPS, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_patients, size=(reps, n_patients))
    W = np.zeros((reps, n_patients))
    for r in range(reps):
        W[r] = np.bincount(idx[r], minlength=n_patients)
    return W


def ci(values: np.ndarray) -> list[float] | None:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return None
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


# ---------------------------------------------------------------- one experiment

def run(reference: str, periods, folds: dict[str, int], save_models: bool, with_sensor_off: bool) -> dict:
    t0 = time.time()
    test_rows, test_events, alert_records, event_records, cal_rows, off_frames, thresholds = [], [], [], [], [], [], {}
    exclusions = []
    patients = sorted({p.patient_id for p in periods})
    for k in range(N_OUTER_FOLDS):
        train_pats = {p for p in patients if folds[p] != k}
        prior = calibration_prior(periods, train_pats) if reference == "corrected" else 0.0
        rows_by_role = {"train": [], "test": []}
        off_by_role = {"train": [], "test": []}
        events_test = []
        for p in periods:
            if p.cgm.empty:
                continue
            role = "train" if p.patient_id in train_pats else "test"
            cal = calibrate(p, prior, reference)
            ok, why = eligibility(p, cal)
            if role == "test":
                cal_rows.append({"period": p.period_id, "patient": p.patient_id, "fold": k, "prior": round(prior, 3),
                                 "offset": round(cal.offset, 3), "n_pairs": cal.n_pairs,
                                 "warm_end": cal.warm_end, "cal_end": cal.cal_end, "meal_fill_g": cal.meal_fill,
                                 "n_cal_meals": cal.n_cal_meals, "n_cal_readings": cal.n_cal_readings,
                                 "eligible": ok, "reason": why,
                                 **({f"coef_{c}": v for c, v in cal.twin.coef.items()} if cal.twin else {}),
                                 **(cal.twin.describe() if cal.twin else {})})
                if not ok:
                    exclusions.append({"period": p.period_id, "patient": p.patient_id, "reason": why})
            if not ok:
                continue
            rows, events = period_rows(cal, p)
            rows_by_role[role].append(rows)
            if role == "test":
                events_test.append(events)
            if with_sensor_off:
                frame = sensor_off_frame(cal, p)
                if frame is not None:
                    off_by_role[role].append(frame)
        train = pd.concat(rows_by_role["train"], ignore_index=True)
        test = pd.concat(rows_by_role["test"], ignore_index=True)
        test["fold"] = k
        ev_test = pd.concat(events_test, ignore_index=True) if events_test else pd.DataFrame()

        # Outer models: every arm and horizon, plus the classifier.
        for arm, feats in ARMS.items():
            for h in HORIZONS_MIN:
                ok = train[f"y_{h}"].notna()
                m = fit_lgb(train.loc[ok, feats], train.loc[ok, f"y_{h}"])
                test[f"{arm}_{h}"] = forecast(m, test, feats)
                if save_models and arm == "C":
                    MODELS.mkdir(parents=True, exist_ok=True)
                    m.booster_.save_model(str(MODELS / f"fold{k}_C_{h}.txt"))
        clf = fit_lgb(train[ARMS["C"]], train["low_next60"], binary=True)
        test["p_low"] = low_probability(clf, test, ARMS["C"])
        if save_models:
            clf.booster_.save_model(str(MODELS / f"fold{k}_classifier.txt"))

        # Threshold selection on outer-training patients, from alert rates only. Learned scores are
        # inner out-of-fold scores with their own inner offset prior; the others are outer-training rows.
        inner = inner_learned_scores(periods, train, reference)
        thresholds[k] = {}
        for method, (col, kind) in ALERT_METHODS.items():
            thresholds[k][method] = {}
            frame = inner if col in ("B_30", "C_30", "p_low") else train
            for budget in ALERT_BUDGETS_PER_DAY:
                thresholds[k][method][f"budget_{budget}"] = select_threshold(frame, col, kind, budget)
            if kind == "glucose":
                thresholds[k][method]["fixed_85"] = 85.0
                thresholds[k][method]["fixed_90"] = 90.0

        # Emit and score alerts on the held-out patients.
        for method, (col, kind) in ALERT_METHODS.items():
            for setting, thr in thresholds[k][method].items():
                policy = AlertPolicy(thr, kind)
                for period, g in test.groupby("period", sort=False):
                    g = g.sort_values("time")
                    fired = emit(g["time"], g[col], g["g0"], policy)
                    pev = ev_test[ev_test["period"] == period] if len(ev_test) else ev_test
                    readings = pd.DatetimeIndex(g["time"])
                    evr, alr = score_alerts(g["time"][fired], pev["event_start"] if len(pev) else [],
                                            pev["eligible"] if len(pev) else [], readings)
                    pat = g["patient"].iloc[0]
                    for r in evr:
                        event_records.append({"method": method, "setting": setting, "period": period, "patient": pat, **r})
                    for r in alr:
                        alert_records.append({"method": method, "setting": setting, "period": period, "patient": pat, **r})
                    alert_records.append({"method": method, "setting": setting, "period": period, "patient": pat,
                                          "alert_time": pd.NaT, "class": "_monitored", "origins": len(g)})
        test_rows.append(test)
        if len(ev_test):
            test_events.append(ev_test)

        # Sensor-off: learning layer fitted on training patients' frames.
        if with_sensor_off and off_by_role["test"]:
            tr = pd.concat(off_by_role["train"], ignore_index=True)
            te = pd.concat(off_by_role["test"], ignore_index=True)
            feats = OFF_FEATURES + EHR
            m = fit_lgb(tr[feats], tr["truth"])
            te["hybrid"] = booster_of(m).predict(te[feats])
            te["fold"] = k
            off_frames.append(te)
        print(f"[{reference}] fold {k}: train rows {len(train)}, test rows {len(test)}, "
              f"{time.time() - t0:.0f}s", flush=True)

    preds = pd.concat(test_rows, ignore_index=True)
    events = pd.concat(test_events, ignore_index=True) if test_events else pd.DataFrame()
    alerts = pd.DataFrame(alert_records)
    evrec = pd.DataFrame(event_records)
    off = pd.concat(off_frames, ignore_index=True) if off_frames else pd.DataFrame()
    result = summarise(preds, events, alerts, evrec, off, thresholds, reference)
    result["elapsed_s"] = round(time.time() - t0, 1)
    return {"result": result, "preds": preds, "events": events, "alerts": alerts, "evrec": evrec,
            "off": off, "calibrations": pd.DataFrame(cal_rows), "exclusions": pd.DataFrame(exclusions)}


# ---------------------------------------------------------------- summaries

def forecast_summary(preds: pd.DataFrame) -> tuple[dict, dict]:
    pats = sorted(preds["patient"].unique())
    pidx = {p: i for i, p in enumerate(pats)}
    W = weights(len(pats))
    out, paired = {}, {}
    for h in HORIZONS_MIN:
        d = preds[preds[f"y_{h}"].notna()].copy()
        truth = (d["g0"] + d[f"y_{h}"]).to_numpy()
        fc = {"persistence": d["g0"].to_numpy(), "trend": d[f"trend_{h}"].to_numpy(),
              "physiology": d[f"phys_{h}"].to_numpy(), **{a: d[f"{a}_{h}"].to_numpy() for a in ("A", "B", "D", "C")}}
        pi = d["patient"].map(pidx).to_numpy()
        n_p = np.bincount(pi, minlength=len(pats)).astype(float)
        stats = {}
        for m, f in fc.items():
            err = f - truth
            sse = np.bincount(pi, weights=err ** 2, minlength=len(pats))
            are = np.bincount(pi, weights=np.abs(err) / truth, minlength=len(pats))
            ab = np.isin(clarke_zones(truth, f), ["A", "B"]).astype(float)
            abn = np.bincount(pi, weights=ab, minlength=len(pats))
            boot_rmse = np.sqrt((W @ sse) / (W @ n_p))
            stats[m] = boot_rmse
            out.setdefault(f"{h} min", {})[m] = {
                "name": METHOD_NAMES[m], "rmse_mg_dl": round(float(np.sqrt(sse.sum() / n_p.sum())), 4),
                "rmse_95ci": ci(boot_rmse), "mard_percent": round(float(are.sum() / n_p.sum() * 100), 2),
                "clarke_a_plus_b_percent": round(float(abn.sum() / n_p.sum() * 100), 2)}
        out[f"{h} min"]["points"] = int(len(d))
        out[f"{h} min"]["patients"] = int(d["patient"].nunique())
        comps = {"C minus B (value of the physiology core)": ("C", "B"),
                 "C minus D (value of the clinical record, with physiology)": ("C", "D"),
                 "B minus A (value of the clinical record, without physiology)": ("B", "A"),
                 "D minus A (value of the physiology core, without record)": ("D", "A"),
                 "B minus persistence": ("B", "persistence"), "C minus persistence": ("C", "persistence"),
                 "trend minus persistence": ("trend", "persistence"),
                 "physiology minus persistence": ("physiology", "persistence")}
        paired[f"{h} min"] = {}
        for label, (a, b) in comps.items():
            diff = stats[a] - stats[b]
            point = out[f"{h} min"][a]["rmse_mg_dl"] - out[f"{h} min"][b]["rmse_mg_dl"]
            paired[f"{h} min"][label] = {"rmse_difference_mg_dl": round(point, 4), "95ci": ci(diff),
                                         "share_of_resamples_below_zero": round(float(np.mean(diff < 0)), 3)}
    return out, paired


def alarm_summary(alerts: pd.DataFrame, evrec: pd.DataFrame, events: pd.DataFrame) -> dict:
    out = {}
    pats = sorted(alerts["patient"].unique())
    pidx = {p: i for i, p in enumerate(pats)}
    W = weights(len(pats), seed=SEED + 2)
    for (method, setting), a in alerts.groupby(["method", "setting"], sort=False):
        e = evrec[(evrec["method"] == method) & (evrec["setting"] == setting)]
        mon = a[a["class"] == "_monitored"]
        real = a[a["class"] != "_monitored"]

        def per_patient(frame, mask=None, col=None):
            f = frame if mask is None else frame[mask]
            vals = np.zeros(len(pats))
            if len(f):
                s = f.groupby("patient").size() if col is None else f.groupby("patient")[col].sum()
                for p, v in s.items():
                    vals[pidx[p]] = v
            return vals

        ev_n = per_patient(e)
        caught = per_patient(e, e["caught"].astype(bool)) if len(e) else np.zeros(len(pats))
        days = per_patient(mon, col="origins") * GRID_MIN / 1440.0
        cls = {c: per_patient(real, real["class"] == c) for c in ("matched", "redundant", "false", "censored")}
        useful = cls["matched"] + cls["redundant"]
        b_ev = W @ ev_n
        b_sens = np.where(b_ev > 0, (W @ caught) / np.where(b_ev > 0, b_ev, 1), np.nan)
        b_prec_den = W @ (useful + cls["false"])
        b_prec = np.where(b_prec_den > 0, (W @ useful) / np.where(b_prec_den > 0, b_prec_den, 1), np.nan)
        b_fpd = (W @ cls["false"]) / (W @ days)
        leads = e.loc[e["caught"].astype(bool), "lead_min"] if len(e) else pd.Series(dtype=float)
        n_ev, n_c = int(ev_n.sum()), int(caught.sum())
        emitted = int(sum(v.sum() for v in cls.values()))
        prec_den = useful.sum() + cls["false"].sum()
        out.setdefault(setting, {})[method] = {
            "name": ALERT_NAMES[method], "eligible_events": n_ev,
            "patients_with_eligible_events": int((ev_n > 0).sum()), "caught": n_c,
            "sensitivity_percent": round(100 * n_c / n_ev, 1) if n_ev else None,
            "sensitivity_95ci": [round(100 * x, 1) for x in ci(b_sens)] if ci(b_sens) else None,
            "bootstrap_resamples_with_no_event": int(np.sum(~np.isfinite(b_sens))),
            "emitted_alerts": emitted, "matched": int(cls["matched"].sum()), "redundant": int(cls["redundant"].sum()),
            "false": int(cls["false"].sum()), "censored": int(cls["censored"].sum()),
            "alert_precision_percent": round(100 * useful.sum() / prec_den, 1) if prec_den else None,
            "alert_precision_95ci": [round(100 * x, 1) for x in ci(b_prec)] if ci(b_prec) else None,
            "monitored_patient_days": round(float(days.sum()), 1),
            "emitted_alerts_per_day": round(emitted / days.sum(), 3) if days.sum() else None,
            "false_alerts_per_day": round(float(cls["false"].sum() / days.sum()), 4) if days.sum() else None,
            "false_alerts_per_day_95ci": ci(b_fpd),
            "lead_min_median": round(float(leads.median()), 1) if len(leads) else None,
            "lead_min_iqr": [round(float(leads.quantile(0.25)), 1), round(float(leads.quantile(0.75)), 1)]
            if len(leads) else None}
    return out


def classifier_summary(preds: pd.DataFrame) -> dict:
    d = preds[preds["g0"] >= LOW_MG_DL]
    y, p = d["low_next60"].to_numpy(), d["p_low"].to_numpy()
    pats = sorted(d["patient"].unique())
    rng = np.random.default_rng(SEED + 3)
    by = {q: g for q, g in d.groupby("patient")}
    aucs, aps = [], []
    for _ in range(300):
        pick = rng.choice(pats, size=len(pats), replace=True)
        s = pd.concat([by[q] for q in pick])
        if s["low_next60"].nunique() < 2:
            continue
        aucs.append(roc_auc_score(s["low_next60"], s["p_low"]))
        aps.append(average_precision_score(s["low_next60"], s["p_low"]))
    return {"windows": int(len(d)), "windows_with_low_within_60_min": int(y.sum()),
            "prevalence": round(float(y.mean()), 4), "auroc": round(float(roc_auc_score(y, p)), 3),
            "auroc_95ci": ci(np.array(aucs)), "auprc": round(float(average_precision_score(y, p)), 3),
            "auprc_95ci": ci(np.array(aps))}


def sensor_off_summary(off: pd.DataFrame) -> dict:
    if off.empty:
        return {}
    pats = sorted(off["patient"].unique())
    pidx = {p: i for i, p in enumerate(pats)}
    W = weights(len(pats), seed=SEED + 4)
    pi = off["patient"].map(pidx).to_numpy()
    n_p = np.bincount(pi, minlength=len(pats)).astype(float)
    truth = off["truth"].to_numpy()
    res = {"periods": int(off["period"].nunique()), "patients": len(pats), "points": int(len(off)),
           "hours": round(len(off) * GRID_MIN / 60, 1)}
    days = off.groupby("period").size() * GRID_MIN / 1440
    res["days_scored_per_period"] = {"median": round(float(days.median()), 2), "max": round(float(days.max()), 2)}
    boots = {}
    for m in ("carried", "typical", "twin", "hybrid"):
        err = off[m].to_numpy() - truth
        sse = np.bincount(pi, weights=err ** 2, minlength=len(pats))
        boots[m] = np.sqrt((W @ sse) / (W @ n_p))
        low = truth < LOW_MG_DL
        res[m] = {"rmse_mg_dl": round(float(np.sqrt(np.mean(err ** 2))), 4), "rmse_95ci": ci(boots[m]),
                  "mard_percent": round(float(np.mean(np.abs(err) / truth) * 100), 2),
                  "clarke_a_plus_b_percent": round(float(np.isin(clarke_zones(truth, off[m]), ["A", "B"]).mean() * 100), 2),
                  "rmse_when_reference_below_70": round(float(np.sqrt(np.mean(err[low] ** 2))), 2) if low.any() else None,
                  "share_of_readings_below_70_estimated_below_80_percent":
                      round(float((off.loc[low, m] < 80).mean() * 100), 1) if low.any() else None}
    res["readings_below_70"] = int((truth < LOW_MG_DL).sum())
    res["hybrid_minus_carried_rmse_95ci"] = ci(boots["hybrid"] - boots["carried"])
    res["hybrid_minus_typical_rmse_95ci"] = ci(boots["hybrid"] - boots["typical"])
    bins = pd.cut(off["hours_since_anchor"], [0, 2, 4, 8, 12, 48, 1000], include_lowest=True)
    res["by_hours_since_anchor"] = [
        {"hours": str(b), "points": int(len(g)),
         **{f"{m}_rmse": round(float(np.sqrt(np.mean((g[m] - g["truth"]) ** 2))), 1)
            for m in ("carried", "typical", "twin", "hybrid")}}
        for b, g in off.groupby(bins, observed=True)]
    return res


def summarise(preds, events, alerts, evrec, off, thresholds, reference) -> dict:
    forecast, paired = forecast_summary(preds)
    res = {"reference": reference,
           "data": {"periods_scored": int(preds["period"].nunique()), "patients_scored": int(preds["patient"].nunique()),
                    "forecast_origins": int(len(preds)),
                    "monitored_patient_days": round(len(preds) * GRID_MIN / 1440.0, 1),
                    "events_after_calibration": int(len(events)),
                    "eligible_events": int(events["eligible"].sum()) if len(events) else 0,
                    "patients_with_eligible_events": int(events.loc[events["eligible"], "patient"].nunique())
                    if len(events) else 0},
           "forecast": forecast, "paired_differences": paired,
           "alarms": alarm_summary(alerts, evrec, events), "classifier": classifier_summary(preds),
           "thresholds_by_fold": thresholds}
    if not off.empty:
        res["sensor_off"] = sensor_off_summary(off)
    return res


# ---------------------------------------------------------------- main

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    t0 = time.time()
    code_files = sorted((ROOT / "twin").glob("*.py")) + [ROOT / "scripts" / "evaluate.py"]
    code_at_start = {str(f.relative_to(ROOT)): sha256(f) for f in code_files}  # hashed before the run
    periods = load_all()
    patients = sorted({p.patient_id for p in periods})
    folds = assign_folds(patients)
    OUT.mkdir(exist_ok=True)
    (OUT / "folds.json").write_text(json.dumps({"seed": SEED, "n_folds": N_OUTER_FOLDS, "patient_fold": folds}, indent=1))

    main_run = run("corrected", periods, folds, save_models=True, with_sensor_off=True)
    raw_run = run("raw", periods, folds, save_models=False, with_sensor_off=False)

    result = main_run["result"]
    result["protocol"] = {"version": PROTOCOL_VERSION, "document": "docs/evaluation_protocol.md",
                          "sha256": sha256(ROOT / "docs" / "evaluation_protocol.md")}
    result["excluded_periods"] = int(len(main_run["exclusions"]))
    (OUT / "evaluation.json").write_text(json.dumps(result, indent=1, default=str))
    raw = raw_run["result"]
    raw["protocol"] = result["protocol"]
    (OUT / "evaluation_raw.json").write_text(json.dumps(raw, indent=1, default=str))

    keep = ["period", "patient", "fold", "pos", "time", "g0", "low_next60", "p_low"] + \
           [f"y_{h}" for h in HORIZONS_MIN] + [f"trend_{h}" for h in HORIZONS_MIN] + \
           [f"phys_{h}" for h in HORIZONS_MIN] + [f"{a}_{h}" for a in ("A", "B", "D", "C") for h in HORIZONS_MIN]
    main_run["preds"][keep].to_csv(OUT / "predictions.csv.gz", index=False)
    main_run["alerts"].to_csv(OUT / "alerts.csv.gz", index=False)
    main_run["events"].to_csv(OUT / "events.csv", index=False)
    main_run["exclusions"].to_csv(OUT / "exclusions.csv", index=False)
    cals = main_run["calibrations"]
    cals.to_csv(OUT / "calibrations.csv", index=False)
    twins = cals[cals["eligible"]].copy()
    twins.to_csv(OUT / "twins.csv", index=False)
    if not main_run["off"].empty:
        main_run["off"].to_csv(OUT / "sensor_off.csv.gz", index=False)
    schema = {"features": ARMS["C"], "classifier_features": ARMS["C"], "target": "change in mg/dL at the horizon",
              "horizons_min": list(HORIZONS_MIN), "missing_values": "NaN, handled natively by LightGBM",
              "fold_rule": "fold k model was trained on every patient NOT in fold k (results/folds.json)",
              "thresholds_by_fold": result["thresholds_by_fold"]}
    (MODELS / "schema.json").write_text(json.dumps(schema, indent=1))

    data_files = sorted((ROOT / "data" / "shanghai_t2dm").rglob("*.xls*"))
    changed = [f for f, h in code_at_start.items() if sha256(ROOT / f) != h]
    if changed:
        print("WARNING: code changed during the run:", changed)
    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(), "protocol": result["protocol"], "seed": SEED,
        "python": platform.python_version(), "platform": platform.platform(),
        "libraries": {lib: md.version(lib) for lib in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm")},
        "data_sha256": {str(f.relative_to(ROOT)): sha256(f) for f in data_files},
        "code_sha256": code_at_start, "code_changed_during_run": changed,
        "config": {"lgb_params": LGB_PARAMS, "alert_budgets_per_day": list(ALERT_BUDGETS_PER_DAY),
                   "primary_budget": PRIMARY_BUDGET, "cooldown_min": ALERT_COOLDOWN_MIN,
                   "warning_window_min": WARNING_WINDOW_MIN, "bootstrap_reps": BOOTSTRAP_REPS},
        "exclusions": main_run["exclusions"].to_dict(orient="records"),
        "elapsed_s": round(time.time() - t0, 1)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(json.dumps({k: result[k] for k in ("data", "forecast")}, indent=1, default=str))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
