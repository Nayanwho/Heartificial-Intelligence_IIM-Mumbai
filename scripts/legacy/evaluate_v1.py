"""SUPERSEDED (version 1, 19 October 2026). Kept only to reproduce results/legacy_2026-10-19/.

This evaluation lets later data into earlier inputs: the sensor offset uses the whole period's
fingersticks, the offset prior includes test patients, missing meals use a whole-period median,
the calibration length depends on the period length, and sensor-off mode starts from a hidden
CGM reading. Use scripts/evaluate.py (version 2, docs/evaluation_protocol.md) instead.

Original description: evaluate the twin on real patients: forecasts, low alarms, and the CGM-off test.

Protocol
- Each sensor period is split in time: the twin is fitted on the first 5 days after
  the warm-up day (at most 60% of the period) and tested on the rest.
- Truth is the sensor reading corrected for that person's offset (see twin/sensor.py).
- Forecasts made at time t use only doses and meals logged up to t.
- The learning layer is trained across patients and always tested on patients it has
  not seen (5-fold split by patient).

Run from the repository root:  python scripts/legacy/evaluate_v1.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from twin.data_shanghai import load_all  # noqa: E402
from twin.inputs import InputBank  # noqa: E402
from twin.metrics import alarm_episodes, bootstrap_ci, clarke_ab, mard, rmse  # noqa: E402
from twin.model import fit_twin, simulate_array  # noqa: E402
from twin.sensor import LOW_MG_DL, check_sensor, low_events, pair_fingersticks, trim_warmup  # noqa: E402

HORIZONS = {30: 2, 60: 4, 120: 8}
CAL_DAYS, CAL_FRACTION, MIN_TEST_DAYS = 5.0, 0.6, 1.0
OUT = ROOT / "results" / "legacy_2026-10-19"
SEED = 7


def split_point(g: pd.Series) -> pd.Timestamp | None:
    days = (g.index.max() - g.index.min()).total_seconds() / 86400
    cal_end = g.index.min() + pd.Timedelta(days=min(CAL_DAYS, CAL_FRACTION * days))
    if (g.index.max() - cal_end).total_seconds() / 86400 < MIN_TEST_DAYS:
        return None
    return cal_end


def static_features(p) -> dict:
    info = p.info
    classes = p.drug_classes()
    return {
        "age": info.get("age"), "bmi": info.get("bmi"), "hba1c": info.get("hba1c_percent"),
        "egfr": info.get("egfr"), "duration_years": info.get("duration_years"),
        "c_peptide": info.get("c_peptide_fasting"), "male": 1.0 if info.get("sex") == "M" else 0.0,
        "on_su": float("sulfonylurea" in classes), "on_insulin": float("insulin" in classes),
    }


def build_rows(periods, prior: float):
    rows, twins, coast = [], [], []
    for n, p in enumerate(periods):
        chk = check_sensor(p.cgm, p.fingerstick, prior)
        g = chk.corrected
        if len(g) < 300:
            continue
        cal_end = split_point(g)
        if cal_end is None:
            continue
        bank = InputBank(p, g.index)
        full = bank.full()
        cal = g[g.index < cal_end]
        try:
            twin = fit_twin(cal, full.loc[cal.index])
        except ValueError:
            continue
        twins.append({"period": p.period_id, "patient": p.patient_id, "offset": chk.offset,
                      **twin.describe(), **{f"coef_{k}": v for k, v in twin.coef.items()}})

        times = g.index
        gv = g.to_numpy()
        minutes = np.r_[0.0, np.diff(times.values).astype("timedelta64[s]").astype(float) / 60.0]
        events = low_events(g, LOW_MG_DL)
        starts = pd.DatetimeIndex(events["start"]) if len(events) else pd.DatetimeIndex([])
        static = static_features(p)
        test_pos = np.where(times >= cal_end)[0]
        full_arr = full[["insulin", "iob", "su", "glp1", "carbs", "cob"]].to_numpy()
        for i in test_pos:
            if i < 8 or i + 8 >= len(g):
                continue
            idx = np.arange(i, i + 9)
            X = bank.known_array(i, idx)
            sim = simulate_array(twin, gv[i], X, minutes[idx])
            t = times[i]
            row = {"period": p.period_id, "patient": p.patient_id, "time": t, "g0": gv[i],
                   "lag15": gv[i - 1], "lag30": gv[i - 2], "lag60": gv[i - 4], "lag120": gv[i - 8],
                   "hour_sin": np.sin(2 * np.pi * (t.hour + t.minute / 60) / 24),
                   "hour_cos": np.cos(2 * np.pi * (t.hour + t.minute / 60) / 24),
                   "insulin_now": full_arr[i, 0], "iob": full_arr[i, 1], "su_now": full_arr[i, 2],
                   "glp1_now": full_arr[i, 3], "carbs_now": full_arr[i, 4], "cob": full_arr[i, 5],
                   "sd_2h": float(np.std(gv[i - 8:i + 1])), "offset": chk.offset, **static}
            for mins, h in HORIZONS.items():
                row[f"twin_{mins}"] = sim[h] - gv[i]
                gap = (times[i + h] - t).total_seconds() / 60
                row[f"y_{mins}"] = gv[i + h] - gv[i] if abs(gap - mins) <= 10 else np.nan
            row["twin_min_120"] = float(sim.min()) - gv[i]
            ahead = (starts > t) & (starts <= t + pd.Timedelta(minutes=60))
            row["low_next60"] = int(ahead.any())
            rows.append(row)

        # CGM-off test: from the end of calibration the twin sees only fingersticks, doses and meals.
        fs = p.fingerstick[(p.fingerstick.index >= cal_end) & (p.fingerstick.index <= times.max())]
        fs = fs[~fs.index.duplicated(keep="last")]
        test = g[g.index >= cal_end]
        if len(fs) >= 3 and (test.index.max() - test.index.min()).total_seconds() / 86400 >= 1.5:
            start_i = int(np.searchsorted(times.values, np.datetime64(cal_end)))
            sim = twin.simulate(gv[start_i], full.iloc[start_i:], anchors=fs)
            carried = fs.reindex(test.index, method="ffill").fillna(gv[start_i])
            profile = cal.groupby(cal.index.hour).mean()
            typical = pd.Series(test.index.hour.map(profile).to_numpy(), index=test.index)
            last_fs_time = pd.Series(fs.index, index=fs.index).reindex(test.index, method="ffill")
            hours_since = (test.index.to_series() - last_fs_time).dt.total_seconds() / 3600
            last_fs_value = fs.reindex(test.index, method="ffill")
            inputs_now = full.loc[test.index]
            frame = pd.DataFrame({"period": p.period_id, "patient": p.patient_id, "truth": test,
                                  "twin": sim.reindex(test.index), "carried": carried,
                                  "typical": typical, "hours_since_fingerstick": hours_since,
                                  "last_fingerstick": last_fs_value, "cal_mean": float(cal.mean()),
                                  "cal_sd": float(cal.std()), "iob": inputs_now["iob"], "cob": inputs_now["cob"],
                                  "su_now": inputs_now["su"], "insulin_now": inputs_now["insulin"],
                                  "carbs_now": inputs_now["carbs"],
                                  "hour_sin": np.sin(2 * np.pi * test.index.hour / 24),
                                  "hour_cos": np.cos(2 * np.pi * test.index.hour / 24)})
            for k, v in static.items():
                frame[k] = v
            coast.append(frame)
        print(f"{n + 1}/{len(periods)} {p.period_id}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(twins), (pd.concat(coast) if coast else pd.DataFrame())


BASE_FEATURES = ["g0", "lag15", "lag30", "lag60", "lag120", "hour_sin", "hour_cos", "insulin_now", "iob",
                 "su_now", "glp1_now", "carbs_now", "cob", "sd_2h", "offset", "age", "bmi", "hba1c", "egfr",
                 "duration_years", "c_peptide", "male", "on_su", "on_insulin"]
TWIN_FEATURES = ["twin_30", "twin_60", "twin_120", "twin_min_120"]


def oof_predict(df: pd.DataFrame, features: list[str], target: str, objective: str) -> np.ndarray:
    """Out-of-fold predictions: every row is predicted by a model that never saw its patient."""
    pred = np.full(len(df), np.nan)
    data = df.dropna(subset=[target])
    folds = GroupKFold(n_splits=5)
    params = dict(objective=objective, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_estimators=400,
                  random_state=SEED, verbose=-1)
    for train, test in folds.split(data, groups=data["patient"]):
        model = lgb.LGBMRegressor(**params) if objective != "binary" else lgb.LGBMClassifier(**params)
        model.fit(data.iloc[train][features], data.iloc[train][target])
        out = model.predict(data.iloc[test][features]) if objective != "binary" else \
            model.predict_proba(data.iloc[test][features])[:, 1]
        pred[data.index[test]] = out
    return pred


def forecast_metrics(df: pd.DataFrame) -> dict:
    result = {}
    for mins in HORIZONS:
        target = f"y_{mins}"
        d = df.dropna(subset=[target]).copy()
        d["truth"] = d["g0"] + d[target]
        d["persistence"] = d["g0"]
        d["twin"] = d["g0"] + d[f"twin_{mins}"]
        d["ml"] = d["g0"] + df.loc[d.index, f"ml_{mins}"]
        d["hybrid"] = d["g0"] + df.loc[d.index, f"hybrid_{mins}"]
        result[f"{mins} min"] = {}
        for method in ("persistence", "twin", "ml", "hybrid"):
            stats = {"rmse_mg_dl": round(rmse(d[method], d["truth"]), 1),
                     "mard_percent": round(mard(d[method], d["truth"]), 1),
                     "clarke_a_plus_b_percent": round(clarke_ab(d[method], d["truth"]), 1)}
            if method in ("persistence", "hybrid"):
                lo, hi = bootstrap_ci(d, lambda s, m=method: rmse(s[m], s["truth"]), n=300)
                stats["rmse_95ci"] = [round(lo, 1), round(hi, 1)]
            result[f"{mins} min"][method] = stats
        result[f"{mins} min"]["points"] = int(len(d))
        result[f"{mins} min"]["patients"] = int(d["patient"].nunique())
    return result


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (centre - half), 1), round(100 * (centre + half), 1)]


def alarm_rule(d: pd.DataFrame, events_by_period: dict, alarm_col: str, test_days: float) -> dict:
    """Event-level check: a low counts as caught if an alarm sounded in the hour before it began."""
    detected, leads, false_eps, total = 0, [], 0, 0
    for period, g in d.groupby("period"):
        g = g.sort_values("time")
        ev = [e for e in events_by_period.get(period, [])
              if g["time"].min() + pd.Timedelta(minutes=60) <= e <= g["time"].max()]
        total += len(ev)
        for e in ev:
            window = g[(g["time"] >= e - pd.Timedelta(minutes=60)) & (g["time"] < e)]
            hits = window[window[alarm_col]]
            if len(hits):
                detected += 1
                leads.append((e - hits["time"].min()).total_seconds() / 60)
        for s in alarm_episodes(g["time"], g[alarm_col].to_numpy()):
            if not any(s <= e <= s + pd.Timedelta(minutes=75) for e in ev):
                false_eps += 1
    return {"lows": total, "caught": detected, "sensitivity_percent": round(100 * detected / max(total, 1), 1),
            "sensitivity_95ci": wilson(detected, total),
            "median_warning_min": float(np.median(leads)) if leads else None,
            "false_alarms_per_day": round(false_eps / test_days, 2)}


def low_metrics(df: pd.DataFrame, events_by_period: dict) -> dict:
    d = df[df["g0"] >= LOW_MG_DL].copy()
    y, p = d["low_next60"].to_numpy(), d["p_low"].to_numpy()
    test_days = float(d.groupby("period")["time"].agg(lambda s: (s.max() - s.min()).total_seconds() / 86400).sum())
    res = {"windows": int(len(d)), "windows_with_low_ahead": int(y.sum()), "test_days": round(test_days, 1),
           "classifier_auroc": round(float(roc_auc_score(y, p)), 3),
           "classifier_auprc": round(float(average_precision_score(y, p)), 3), "base_rate": round(float(y.mean()), 4)}
    lo, hi = bootstrap_ci(d, lambda s: roc_auc_score(s["low_next60"], s["p_low"]), n=300)
    res["classifier_auroc_95ci"] = [round(lo, 3), round(hi, 3)]
    d["forecast30"] = d["g0"] + d["hybrid_30"]
    d["forecast60"] = d["g0"] + d["hybrid_60"]
    rules = {}
    for name, col, thr in [("forecast 30 min ahead <= 85", "forecast30", 85),
                           ("forecast 30 min ahead <= 90", "forecast30", 90),
                           ("forecast 60 min ahead <= 95", "forecast60", 95)]:
        d["alarm"] = d[col] <= thr
        rules[name] = alarm_rule(d, events_by_period, "alarm", test_days)
    d["alarm"] = d["p_low"] >= 0.001
    rules["classifier probability >= 0.001"] = alarm_rule(d, events_by_period, "alarm", test_days)
    res["alarm_rules"] = rules
    return res


COAST_FEATURES = ["twin", "typical", "carried", "last_fingerstick", "hours_since_fingerstick", "cal_mean",
                  "cal_sd", "iob", "cob", "su_now", "insulin_now", "carbs_now", "hour_sin", "hour_cos", "age",
                  "bmi", "hba1c", "egfr", "duration_years", "c_peptide", "male", "on_su", "on_insulin"]


def coast_hybrid(c: pd.DataFrame) -> pd.DataFrame:
    """The learning layer for sensor-off mode: combines the twin, the person's learned day pattern
    and the last fingerstick. Trained across patients; every patient is predicted by a model that
    never saw them."""
    c = c.dropna(subset=["truth", "twin", "carried", "typical"]).reset_index(drop=True)
    c["twin_minus_typical"] = c["twin"] - c["typical"]
    feats = COAST_FEATURES + ["twin_minus_typical"]
    c["hybrid"] = oof_predict(c, feats, "truth", "regression")
    return c


def coast_metrics(c: pd.DataFrame) -> dict:
    """`c` must already hold the learning layer's predictions (see coast_hybrid)."""
    out = {"periods": int(c["period"].nunique()), "patients": int(c["patient"].nunique()), "points": int(len(c)),
           "hours": round(len(c) * 0.25, 1)}
    for method in ("carried", "typical", "twin", "hybrid"):
        out[method] = {"rmse_mg_dl": round(rmse(c[method], c["truth"]), 1),
                       "mard_percent": round(mard(c[method], c["truth"]), 1),
                       "clarke_a_plus_b_percent": round(clarke_ab(c[method], c["truth"]), 1)}
    lo, hi = bootstrap_ci(c, lambda s: rmse(s["hybrid"], s["truth"]) - rmse(s["carried"], s["truth"]), n=300)
    out["hybrid_minus_carried_rmse_95ci"] = [round(lo, 1), round(hi, 1)]
    lows = c[c["truth"] < LOW_MG_DL]
    out["readings_below_70"] = int(len(lows))
    out["below_70_flagged_under_80_percent"] = {m: round(float((lows[m] < 80).mean() * 100), 1)
                                                for m in ("carried", "typical", "twin", "hybrid")}
    bins = pd.cut(c["hours_since_fingerstick"], [0, 2, 4, 8, 12, 48], include_lowest=True)
    by = []
    for b, g in c.groupby(bins, observed=True):
        by.append({"hours_since_fingerstick": str(b), "points": int(len(g)),
                   **{f"{m}_rmse": round(rmse(g[m], g["truth"]), 1)
                      for m in ("carried", "typical", "twin", "hybrid")}})
    out["by_hours_since_fingerstick"] = by
    return out


def main():
    t0 = time.time()
    periods = load_all()
    prior = float(pd.concat([pair_fingersticks(trim_warmup(p.cgm), p.fingerstick)
                             for p in periods])["diff"].median())
    df, twins, coast = build_rows(periods, prior)
    df = df.reset_index(drop=True)
    print(f"rows {len(df)} in {time.time() - t0:.0f}s", flush=True)

    for mins in HORIZONS:
        df[f"ml_{mins}"] = oof_predict(df, BASE_FEATURES, f"y_{mins}", "regression")
        df[f"hybrid_{mins}"] = oof_predict(df, BASE_FEATURES + TWIN_FEATURES, f"y_{mins}", "regression")
    df["p_low"] = oof_predict(df, BASE_FEATURES + TWIN_FEATURES, "low_next60", "binary")

    events_by_period = {}
    for p in periods:
        chk = check_sensor(p.cgm, p.fingerstick, prior)
        ev = low_events(chk.corrected, LOW_MG_DL)
        events_by_period[p.period_id] = list(pd.DatetimeIndex(ev["start"])) if len(ev) else []

    OUT.mkdir(exist_ok=True)
    coast = coast_hybrid(coast) if len(coast) else coast
    results = {"forecast": forecast_metrics(df), "lows": low_metrics(df, events_by_period),
               "cgm_off": coast_metrics(coast) if len(coast) else {},
               "periods_evaluated": int(df["period"].nunique()), "patients_evaluated": int(df["patient"].nunique()),
               "prior_offset_mg_dl": round(prior, 1)}
    (OUT / "evaluation.json").write_text(json.dumps(results, indent=2, default=str))
    twins.to_csv(OUT / "twins.csv", index=False)
    keep = ["period", "patient", "time", "g0", "y_30", "y_60", "y_120", "twin_30", "twin_60", "twin_120",
            "hybrid_30", "hybrid_60", "hybrid_120", "low_next60", "p_low"]
    df[keep].to_csv(OUT / "cache_predictions.csv.gz", index=False)
    if len(coast):
        coast.to_csv(OUT / "cache_cgm_off.csv.gz", index=False)
    print(json.dumps(results, indent=2, default=str))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
