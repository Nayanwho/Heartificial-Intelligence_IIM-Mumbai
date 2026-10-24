"""A small synthetic Indian cohort for the demo, in the formats a real deployment would read.

Each synthetic person gets:
- a FHIR R4 bundle shaped after India's ABDM implementation guide (v6.5.0): patient, diabetes
  diagnosis, labs (HbA1c, eGFR, creatinine), weight history, CYP2C9 genotype, G6PD status and
  medicines with start dates;
- 14 days of smartwatch and CGM data in Android Health Connect record shapes: heart rate, steps,
  sleep stages, weight and interstitial glucose.

How physiology is assigned: each persona's coefficients are COPIED from the physiology core
fitted (on its 5-day calibration interval, version-2 protocol) to one real ShanghaiT2DM
patient with a clear sulfonylurea effect, named in `seed_period`. Names, ages, routines,
medicines, labs, genetics, fasting plans and all Indian details are synthetic. A persona's
own response to a dose change was never measured. Synthetic data drives the demo only and
never grades a model. The FHIR and Health Connect files are example record shapes; they have
not been validated against the ABDM profiles.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .model import Twin
from dataclasses import replace

from .whatif import DEMO_AS_OF, BodyChange, Routine, Scenario, rehearse

PROFILE = "https://nrces.in/ndhm/fhir/r4/StructureDefinition/"
TODAY = DEMO_AS_OF  # the demo's "today"; kept as a name for older imports
# CYP2C9*3 allele frequency across 36 Indian populations: 11.7% (bioRxiv 113993).
CYP2C9_STAR3 = 0.117
CYP2C9_TESTED_SHARE = 0.4  # demo assumption: most patients are never genotyped
G6PD_TESTED_SHARE = 0.5  # demo assumption
G6PD_DEMO_CASE = "SYN-005"  # one synthetic person is G6PD deficient, to show the safety rule
EXPOSURE_BY_GENOTYPE = {"*1/*1": 1.0, "*1/*3": 1.5, "*3/*3": 2.5, "not tested": 1.0}
FDC_NOTE = ("as a glimepiride 2 mg + metformin 500 mg fixed-dose combination tablet. Twice-daily "
            "glimepiride-metformin combinations are common in Indian practice (real-world study of 7,058 "
            "patients at 347 Indian sites, PMC7920844); the US glimepiride label specifies once daily.")

INDIAN_DAY = [(8.5, 50), (13.5, 70), (17.0, 15), (20.5, 60)]  # breakfast, lunch, tea, dinner (g carbs)


def twin_from_row(row: pd.Series) -> Twin:
    return Twin(coef={k[5:]: float(row[k]) for k in row.index if k.startswith("coef_")}, n_steps=0,
                residual_sd=float(row["residual_sd_mg_dl"]), meta={"seed_period": row["period"]})


def _genotype(rng) -> str:
    a = rng.random() < CYP2C9_STAR3
    b = rng.random() < CYP2C9_STAR3
    return {0: "*1/*1", 1: "*1/*3", 2: "*3/*3"}[int(a) + int(b)]


def make_people(twins: pd.DataFrame, n: int = 24, seed: int = 11) -> list[dict]:
    """Synthetic people; the first is the demo persona."""
    rng = np.random.default_rng(seed)
    pool = twins[(twins["coef_su"] < -0.5) & (twins["coef_su"] > -30)].reset_index(drop=True)
    then = str((DEMO_AS_OF - pd.Timedelta(weeks=12)).date())
    recent = str((DEMO_AS_OF - pd.Timedelta(days=3)).date())
    people = [{
        "id": "SYN-001", "persona": "Meena", "age": 62, "sex": "female", "city": "Bengaluru",
        "height_m": 1.55, "weight_kg_12w_ago": 74.0, "weight_kg_now": 70.3, "weight_dates": [then, recent],
        "hba1c_prev": 8.4, "hba1c_now": 7.1, "egfr_prev": 72, "egfr_now": 58, "egfr_dates": [then, recent],
        "cyp2c9": "*1/*1", "g6pd": "normal",
        "medicines": [("glimepiride", 2.0, "twice daily, before breakfast and dinner, " + FDC_NOTE, "2023-03-01"),
                      ("metformin", 500.0, "twice daily, in the same combination tablet", "2023-03-01"),
                      ("semaglutide", 0.5, "once weekly, Sunday", "2026-09-06")],
        "fasting_plan": {"label": "One-day fast (one evening meal)", "dates": ["2026-10-22"], "window": [6, 18]},
        "seed_period": "2045_0_20201216",
    }]
    i = 2
    while len(people) < n:
        row = pool.iloc[int(rng.integers(len(pool)))]
        on_glp1 = rng.random() < 0.35
        weight_now = float(rng.normal(72, 10))
        weight_prev = weight_now * (1 + (rng.uniform(0.04, 0.09) if on_glp1 else rng.uniform(-0.01, 0.02)))
        egfr_now = int(np.clip(rng.normal(70, 18), 25, 110))
        meds = [("glimepiride", float(rng.choice([1, 2, 4])), "once or twice daily", "2022-01-10"),
                ("metformin", 500.0, "twice daily", "2020-05-01")]
        if on_glp1:
            meds.append(("semaglutide", 0.5, "once weekly", str((TODAY - pd.Timedelta(days=int(rng.integers(14, 80)))).date())))
        if row["coef_insulin"] < -0.5 and rng.random() < 0.5:
            meds.append(("insulin glargine", float(rng.integers(8, 20)), "at bedtime", "2024-02-01"))
        pid = f"SYN-{i:03d}"
        kind = str(rng.choice(["none", "one-day", "multi-day-soon", "month-later"], p=[0.5, 0.2, 0.15, 0.15]))
        if kind == "none":
            plan = None
        elif kind == "one-day":
            plan = {"label": "One-day fast", "window": [6, 18],
                    "dates": [str((DEMO_AS_OF + pd.Timedelta(days=int(rng.integers(1, 12)))).date())]}
        elif kind == "multi-day-soon":
            first = DEMO_AS_OF + pd.Timedelta(days=int(rng.integers(2, 10)))
            plan = {"label": "Nine-day fast (daytime)", "window": [6, 18],
                    "dates": [str((first + pd.Timedelta(days=d)).date()) for d in range(9)]}
        else:
            first = DEMO_AS_OF + pd.Timedelta(days=int(rng.integers(60, 120)))
            plan = {"label": "Month-long daytime fast", "window": [5, 19],
                    "dates": [str((first + pd.Timedelta(days=d)).date()) for d in range(30)]}
        tested_geno = rng.random() < CYP2C9_TESTED_SHARE
        g6pd = ("deficient" if pid == G6PD_DEMO_CASE else
                ("normal" if rng.random() < G6PD_TESTED_SHARE else "not tested"))
        people.append({
            "id": pid, "persona": None, "age": int(rng.integers(45, 81)),
            "sex": str(rng.choice(["female", "male"])), "city": str(rng.choice(["Bengaluru", "Mumbai", "Chennai", "Pune", "Delhi"])),
            "height_m": round(float(rng.normal(1.62, 0.08)), 2), "weight_kg_12w_ago": round(weight_prev, 1),
            "weight_kg_now": round(weight_now, 1), "weight_dates": [then, recent],
            "hba1c_prev": round(float(rng.uniform(7.2, 10.5)), 1),
            "hba1c_now": round(float(rng.uniform(6.6, 9.5)), 1), "egfr_prev": int(egfr_now + rng.integers(0, 20)),
            "egfr_now": egfr_now, "egfr_dates": [then, recent],
            "cyp2c9": _genotype(rng) if tested_geno else "not tested", "g6pd": g6pd, "medicines": meds,
            "fasting_plan": plan, "seed_period": row["period"],
        })
        titrated = titrate(people[-1], twin_from_row(row))
        if titrated is None:  # this seed cannot be kept safe and in range on tablets; draw again
            people.pop()
            continue
        people[-1] = titrated
        i += 1
    return people


def titrate(person: dict, twin: Twin) -> dict:
    """Pick the dose a careful doctor would have reached before any body change:
    the plan with under 5% simulated night-low risk that keeps the most time in range."""
    su_plans = [(1.0, "once daily, before breakfast"), (2.0, "once daily, before breakfast"),
                (1.0, "twice daily, before breakfast and dinner"), (4.0, "once daily, before breakfast"),
                (2.0, "twice daily, before breakfast and dinner")]
    others = [m for m in person["medicines"] if m[0] != "glimepiride"]
    start = next((m[3] for m in person["medicines"] if m[0] == "glimepiride"), "2022-01-10")
    baseline = BodyChange(su_exposure=EXPOSURE_BY_GENOTYPE[person["cyp2c9"]])
    best, best_tir = None, -1.0
    for dose, how in su_plans:
        meds = [("glimepiride", dose, how, start)] + [m for m in others if not m[0] == "semaglutide"]
        trial = {**person, "medicines": meds}
        r = rehearse(twin, routine_for(trial), Scenario(), baseline, days=3, n_sims=60)
        if r["night_low_risk"] < 0.05 and r["time_in_range_percent"] > best_tir:
            best, best_tir = (dose, how), r["time_in_range_percent"]
    if best is None or best_tir < 40:
        return None
    dose, how = best
    return {**person, "medicines": [("glimepiride", dose, how, start)] + others}


def routine_for(person: dict) -> Routine:
    tablets, insulin, weekly = [], [], []
    for name, dose, how, _start in person["medicines"]:
        if name in ("glimepiride", "gliclazide"):
            tablets.append((8.0, name, dose))
            if "twice" in how:
                tablets.append((20.0, name, dose))
        elif name == "metformin":
            tablets += [(8.0, name, dose), (20.0, name, dose)]
        elif name == "semaglutide":
            weekly.append((6, 9.0, name, dose))
        elif name.startswith("insulin"):
            insulin.append((22.0, {"long": 1.0}, dose))
    return Routine(meals=INDIAN_DAY, tablets=tablets, insulin=insulin, weekly=weekly)


APPETITE_CUT_ON_GLP1 = 0.3  # assumption
SENSITIVITY_PER_FRACTION_LOST = 3.0  # assumption: 5% weight lost -> 15% more drug effect, capped at 30%
SU_ACTION_FACTOR = {"below 30": 1.8, "30 to 59": 1.4}  # assumption, stand-in for active-metabolite build-up


def body_change_for(person: dict, as_of: pd.Timestamp | None = None, apply_fasting: bool = True) -> BodyChange:
    """Assumed effects of this person's record, as of a date (all stated, adjustable in the dashboard).

    A planned fast changes meals only on its own dates; it is applied when those dates fall in
    the simulated window."""
    loss = max(0.0, 1 - person["weight_kg_now"] / person["weight_kg_12w_ago"])
    as_of = pd.Timestamp(as_of) if as_of is not None else DEMO_AS_OF
    on_glp1 = any(m[0] == "semaglutide" and pd.Timestamp(m[3]) <= as_of for m in person["medicines"])
    egfr = person["egfr_now"]
    plan = person.get("fasting_plan") or {}
    fast_dates = tuple(plan.get("dates", [])) if apply_fasting else ()
    return BodyChange(
        appetite_cut=APPETITE_CUT_ON_GLP1 if on_glp1 else 0.0,
        sensitivity_gain=round(min(SENSITIVITY_PER_FRACTION_LOST * loss, 0.3), 3),
        su_half_life=(SU_ACTION_FACTOR["below 30"] if egfr is not None and egfr < 30 else
                      SU_ACTION_FACTOR["30 to 59"] if egfr is not None and egfr < 60 else 1.0),
        insulin_duration=1.25 if egfr is not None and egfr < 45 else 1.0,
        su_exposure=EXPOSURE_BY_GENOTYPE.get(person.get("cyp2c9", "not tested"), 1.0),
        fasting_hours=tuple(plan.get("window", (6, 18))) if fast_dates else None,
        fasting_dates=fast_dates,
    )


def fhir_bundle(person: dict) -> dict:
    pid = person["id"]

    def ref(kind, n):
        return f"urn:uuid:{pid.lower()}-{kind}-{n}"

    patient = {"resourceType": "Patient", "id": pid, "meta": {"profile": [PROFILE + "Patient"]},
               "identifier": [{"system": "https://example.org/synthetic", "value": pid}],
               "gender": person["sex"], "birthDate": str(TODAY.year - person["age"]),
               "address": [{"city": person["city"], "country": "IN"}]}
    entries = [{"fullUrl": ref("patient", 1), "resource": patient}]
    subject = {"reference": ref("patient", 1)}
    entries.append({"fullUrl": ref("condition", 1), "resource": {
        "resourceType": "Condition", "meta": {"profile": [PROFILE + "Condition"]}, "subject": subject,
        "code": {"coding": [{"system": "http://snomed.info/sct", "code": "44054006",
                             "display": "Diabetes mellitus type 2"}]}}})

    def obs(n, loinc, display, value, unit, when, profile="Observation"):
        return {"fullUrl": ref("obs", n), "resource": {
            "resourceType": "Observation", "meta": {"profile": [PROFILE + profile]}, "status": "final",
            "code": {"coding": [{"system": "http://loinc.org", "code": loinc, "display": display}]},
            "subject": subject, "effectiveDateTime": when,
            "valueQuantity": {"value": value, "unit": unit, "system": "http://unitsofmeasure.org", "code": unit}}}

    then, now = person.get("egfr_dates") or [str((TODAY - pd.Timedelta(weeks=12)).date()),
                                             str((TODAY - pd.Timedelta(days=3)).date())]
    entries += [
        obs(1, "4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", person["hba1c_prev"], "%", then),
        obs(2, "4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", person["hba1c_now"], "%", now),
        obs(3, "62238-1", "GFR/BSA.pred [Volume Rate/Area] in SerPlBld by CKD-EPI", person["egfr_prev"],
            "mL/min/{1.73_m2}", then),
        obs(4, "62238-1", "GFR/BSA.pred [Volume Rate/Area] in SerPlBld by CKD-EPI", person["egfr_now"],
            "mL/min/{1.73_m2}", now),
        obs(5, "29463-7", "Body weight", person["weight_kg_12w_ago"], "kg", then, "ObservationBodyMeasurement"),
        obs(6, "29463-7", "Body weight", person["weight_kg_now"], "kg", now, "ObservationBodyMeasurement"),
    ]
    if person.get("cyp2c9", "not tested") != "not tested":  # untested genotypes are left out, never invented
        entries.append({"fullUrl": ref("obs", 7), "resource": {
            "resourceType": "Observation", "meta": {"profile": [PROFILE + "Observation"]}, "status": "final",
            "code": {"coding": [{"system": "http://loinc.org", "code": "84413-4", "display": "Genotype display name"}],
                     "text": "CYP2C9 genotype"},
            "subject": subject, "effectiveDateTime": then, "valueString": f"CYP2C9 {person['cyp2c9']}"}})
    if person.get("g6pd", "not tested") != "not tested":
        entries.append({"fullUrl": ref("obs", 8), "resource": {
            "resourceType": "Observation", "meta": {"profile": [PROFILE + "Observation"]}, "status": "final",
            "code": {"text": "G6PD status"}, "subject": subject, "effectiveDateTime": then,
            "valueCodeableConcept": {"text": person["g6pd"]}}})
    for n, (name, dose, how, start) in enumerate(person["medicines"], start=1):
        unit = "IU" if name.startswith("insulin") else "mg"
        entries.append({"fullUrl": ref("med", n), "resource": {
            "resourceType": "MedicationStatement", "meta": {"profile": [PROFILE + "MedicationStatement"]},
            "status": "active", "subject": subject, "medicationCodeableConcept": {"text": name},
            "effectivePeriod": {"start": start},
            "dosage": [{"text": f"{dose:g} {unit} {how}"}]}})
    return {"resourceType": "Bundle", "id": f"{pid}-record", "type": "collection",
            "meta": {"lastUpdated": f"{TODAY.date()}T09:00:00+05:30"}, "entry": entries}


def health_connect(person: dict, glucose: pd.Series, seed: int) -> dict:
    """14 days of watch and CGM data in Health Connect record shapes."""
    rng = np.random.default_rng(seed)
    days = pd.date_range(glucose.index.min().normalize(), periods=14, freq="D")
    hr, steps, sleep = [], [], []
    for d in days:
        for h in range(24):
            start = d + pd.Timedelta(hours=h)
            awake = 6 <= h < 23
            count = int(rng.poisson(450 if awake else 5) * (1.6 if h in (7, 18) else 1.0))
            steps.append({"startTime": start.isoformat(), "endTime": (start + pd.Timedelta(hours=1)).isoformat(),
                          "count": count})
            samples = [{"time": (start + pd.Timedelta(minutes=m)).isoformat(),
                        "beatsPerMinute": int(rng.normal(78 if awake else 62, 4))} for m in range(0, 60, 15)]
            hr.append({"startTime": start.isoformat(), "endTime": (start + pd.Timedelta(hours=1)).isoformat(),
                       "samples": samples})
        bed = d + pd.Timedelta(hours=23)
        stages, t = [], bed
        while t < bed + pd.Timedelta(hours=7):
            stage = str(rng.choice(["light", "deep", "rem", "awake"], p=[0.5, 0.2, 0.22, 0.08]))
            dur = pd.Timedelta(minutes=int(rng.integers(20, 70)))
            stages.append({"startTime": t.isoformat(), "endTime": min(t + dur, bed + pd.Timedelta(hours=7)).isoformat(),
                           "stage": stage})
            t += dur
        sleep.append({"startTime": bed.isoformat(), "endTime": (bed + pd.Timedelta(hours=7)).isoformat(),
                      "stages": stages})
    glucose_records = [{"time": t.isoformat(), "level": {"inMilligramsPerDeciliter": round(float(v), 1)},
                        "specimenSource": "interstitial_fluid"} for t, v in glucose.items()]
    return {"person": person["id"], "source": "synthetic, Health Connect record shapes",
            "HeartRateRecord": hr, "StepsRecord": steps, "SleepSessionRecord": sleep,
            "WeightRecord": [{"time": f"{(TODAY - pd.Timedelta(weeks=12)).date()}T08:00:00+05:30",
                              "weight": {"inKilograms": person["weight_kg_12w_ago"]}},
                             {"time": f"{(TODAY - pd.Timedelta(days=3)).date()}T08:00:00+05:30",
                              "weight": {"inKilograms": person["weight_kg_now"]}}],
            "BloodGlucoseRecord": glucose_records}


def write_cohort(twins: pd.DataFrame, out_dir: Path, n: int = 24) -> pd.DataFrame:
    """Write FHIR bundles, Health Connect files and a cohort summary; returns the summary."""
    (out_dir / "fhir").mkdir(parents=True, exist_ok=True)
    (out_dir / "health_connect").mkdir(parents=True, exist_ok=True)
    rows = []
    for k, person in enumerate(make_people(twins, n)):
        seed_row = twins[twins["period"] == person["seed_period"]].iloc[0]
        twin = twin_from_row(seed_row)
        r = rehearse(twin, routine_for(person), Scenario(), body_change_for(person), days=14, n_sims=60,
                     start=str((TODAY - pd.Timedelta(days=14)).date()), seed=k)
        sensor = pd.Series(r["path"] + np.random.default_rng(k).normal(0, 6, len(r["path"])), index=r["times"])
        (out_dir / "fhir" / f"{person['id']}.json").write_text(json.dumps(fhir_bundle(person), indent=1))
        (out_dir / "health_connect" / f"{person['id']}.json").write_text(json.dumps(health_connect(person, sensor, k)))
        rows.append({"id": person["id"], "persona": person["persona"] or "", "age": person["age"], "sex": person["sex"],
                     "city": person["city"], "egfr_now": person["egfr_now"], "egfr_prev": person["egfr_prev"],
                     "weight_change_percent": round(100 * (person["weight_kg_now"] / person["weight_kg_12w_ago"] - 1), 1),
                     "cyp2c9": person["cyp2c9"], "g6pd": person["g6pd"],
                     "fasting_plan": (f"{person['fasting_plan']['label']}: {person['fasting_plan']['dates'][0]}"
                                      + (f" to {person['fasting_plan']['dates'][-1]}" if len(person['fasting_plan']['dates']) > 1 else ""))
                     if person.get("fasting_plan") else "none",
                     "medicines": "; ".join(f"{m[0]} {m[1]:g}{' IU' if m[0].startswith('insulin') else ' mg'} {m[2]} (since {m[3]})" for m in person["medicines"]),
                     "seed_period": person["seed_period"],
                     "physiology_source": f"core fitted to real ShanghaiT2DM period {person['seed_period']}; persona synthetic"})
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "cohort.csv", index=False)
    (out_dir / "people.json").write_text(json.dumps(make_people(twins, n), indent=1))
    return summary


def dose_options(person: dict) -> dict[str, Routine]:
    """The step-down options a doctor would weigh for this person's current routine."""
    base = routine_for(person)
    su = [t for t in base.tablets if t[1] in ("glimepiride", "gliclazide")]
    others = [t for t in base.tablets if t not in su]
    opts = {"Keep current doses": base}
    if su:
        evening = [t for t in su if t[0] >= 18]
        morning = [t for t in su if t[0] < 18]
        if evening:
            opts["Halve the evening sulfonylurea"] = replace(base, tablets=morning + [(h, d, mg / 2) for h, d, mg in evening] + others)
            opts["Stop the evening sulfonylurea"] = replace(base, tablets=morning + others)
        opts["Halve every sulfonylurea dose"] = replace(base, tablets=[(h, d, mg / 2) for h, d, mg in su] + others)
        if person.get("g6pd") == "deficient":
            opts["Stop every sulfonylurea dose (replacement not simulated)"] = replace(base, tablets=others)
    if base.insulin:
        opts["Insulin 20% lower"] = replace(base, insulin=[(h, c, u * 0.8) for h, c, u in base.insulin])
    return opts
