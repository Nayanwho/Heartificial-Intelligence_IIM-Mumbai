"""Day-one data check: how many real lows does ShanghaiT2DM hold once the sensor is checked?

Run from the repository root:  python scripts/data_check.py
Writes results/data_check.json and results/data_check_periods.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from twin.data_shanghai import load_all  # noqa: E402
from twin.sensor import (LOW_MG_DL, SEVERE_LOW_MG_DL, check_sensor, low_events,  # noqa: E402
                         pair_fingersticks, trim_warmup)


def drug_group(classes: set[str]) -> str:
    if "sulfonylurea" in classes:
        return "sulfonylurea"
    if "insulin" in classes:
        return "insulin, no sulfonylurea"
    return "neither"


def main() -> dict:
    periods = load_all()

    # Cohort prior: median fingerstick-minus-sensor gap after the warm-up day.
    all_pairs = pd.concat([pair_fingersticks(trim_warmup(p.cgm), p.fingerstick).assign(period=p.period_id,
                                                                                       patient=p.patient_id)
                           for p in periods], ignore_index=True)
    prior = float(all_pairs["diff"].median())

    rows = []
    for p in periods:
        chk = check_sensor(p.cgm, p.fingerstick, prior)
        raw70, cor70 = low_events(chk.raw, LOW_MG_DL), low_events(chk.corrected, LOW_MG_DL)
        raw54, cor54 = low_events(chk.raw, SEVERE_LOW_MG_DL), low_events(chk.corrected, SEVERE_LOW_MG_DL)
        classes = p.drug_classes()
        rows.append({
            "period": p.period_id, "patient": p.patient_id, "days_after_warmup": round(
                (chk.raw.index.max() - chk.raw.index.min()).total_seconds() / 86400, 2) if len(chk.raw) else 0.0,
            "readings": int(len(chk.raw)), "fingersticks": int(len(p.fingerstick)), "pairs": chk.n_pairs,
            "offset_mg_dl": round(chk.offset, 1), "group": drug_group(classes),
            "sulfonylurea": "sulfonylurea" in classes, "insulin": "insulin" in classes,
            "glp1": "glp1" in classes, "age": p.info.get("age"), "egfr": p.info.get("egfr"),
            "raw_lows": len(raw70), "corrected_lows": len(cor70),
            "raw_severe_lows": len(raw54), "corrected_severe_lows": len(cor54),
            "raw_night_lows": int(raw70["night"].sum()) if len(raw70) else 0,
            "corrected_night_lows": int(cor70["night"].sum()) if len(cor70) else 0,
        })
    df = pd.DataFrame(rows)

    patients = df.groupby("patient").agg(
        group=("group", lambda s: "sulfonylurea" if (s == "sulfonylurea").any()
               else ("insulin, no sulfonylurea" if (s == "insulin, no sulfonylurea").any() else "neither")),
        days=("days_after_warmup", "sum"), raw_lows=("raw_lows", "sum"), corrected_lows=("corrected_lows", "sum"),
        corrected_night_lows=("corrected_night_lows", "sum"), sulfonylurea=("sulfonylurea", "max"),
        insulin=("insulin", "max"), glp1=("glp1", "max"), age=("age", "first"), egfr=("egfr", "first"))

    by_group = {}
    for name, g in patients.groupby("group"):
        days = g["days"].sum()
        by_group[name] = {
            "patients": int(len(g)), "patient_days": round(float(days), 1),
            "raw_lows_per_100_days": round(100 * g["raw_lows"].sum() / days, 1),
            "corrected_lows_per_100_days": round(100 * g["corrected_lows"].sum() / days, 1),
            "patients_with_corrected_low": int((g["corrected_lows"] > 0).sum()),
        }

    pairs = all_pairs.assign(abs_rel=lambda d: (d["diff"].abs() / d["fingerstick"]) * 100)
    sensor_lows = pairs[pairs["cgm"] < LOW_MG_DL]
    bands = {}
    for lo, hi in [(0, 70), (70, 180), (180, 1000)]:
        b = pairs[(pairs["fingerstick"] >= lo) & (pairs["fingerstick"] < hi)]
        bands[f"fingerstick {lo} to {hi}"] = {"pairs": int(len(b)),
                                              "mean_sensor_minus_fingerstick": round(float(-b["diff"].mean()), 1)}

    result = {
        "periods": int(len(df)), "patients": int(df["patient"].nunique()),
        "sensor_readings_after_warmup": int(df["readings"].sum()),
        "sensor_readings_all": int(sum(len(p.cgm) for p in periods)),
        "fingersticks": int(df["fingersticks"].sum()),
        "fingersticks_per_day_median_where_taken": round(float(
            (df["fingersticks"] / (df["days_after_warmup"] + 1)).loc[df["fingersticks"] > 0].median()), 1),
        "periods_with_fingersticks": int((df["fingersticks"] > 0).sum()),
        "patients_on_sulfonylurea": int(patients["sulfonylurea"].sum()),
        "patients_on_insulin": int(patients["insulin"].sum()),
        "patients_on_glp1": int(patients["glp1"].sum()),
        "patients_aged_65_plus": int((pd.to_numeric(patients["age"], errors="coerce") >= 65).sum()),
        "sensor_check": {
            "pairs_within_8_min": int(len(pairs)),
            "mean_sensor_minus_fingerstick_mg_dl": round(float(-pairs["diff"].mean()), 1),
            "mard_percent": round(float(pairs["abs_rel"].mean()), 1),
            "cohort_prior_offset_mg_dl": round(prior, 1),
            "person_offset_median_mg_dl": round(float(df.groupby("patient")["offset_mg_dl"].first().median()), 1),
            "sensor_lows_with_a_fingerstick": int(len(sensor_lows)),
            "of_which_fingerstick_also_below_70": int((sensor_lows["fingerstick"] < LOW_MG_DL).sum()),
            "by_fingerstick_band": bands,
        },
        "lows": {
            "raw_15min_lows": int(df["raw_lows"].sum()),
            "corrected_15min_lows": int(df["corrected_lows"].sum()),
            "raw_severe_lows_below_54": int(df["raw_severe_lows"].sum()),
            "corrected_severe_lows_below_54": int(df["corrected_severe_lows"].sum()),
            "corrected_night_lows": int(df["corrected_night_lows"].sum()),
            "patients_with_corrected_low": int((patients["corrected_lows"] > 0).sum()),
            "by_drug_group": by_group,
        },
        "cgm_off_test_periods_7_days_plus_with_fingersticks": int(
            ((df["days_after_warmup"] >= 6) & (df["fingersticks"] > 0)).sum()),
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "data_check.json").write_text(json.dumps(result, indent=2))
    df.to_csv(out / "data_check_periods.csv", index=False)
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
