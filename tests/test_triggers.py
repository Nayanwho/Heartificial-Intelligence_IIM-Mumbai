"""Clinician-review triggers read the record as of a given date, and fasting is dated."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twin.synthetic import body_change_for, routine_for  # noqa: E402
from twin.triggers import check  # noqa: E402
from twin.whatif import routine_period  # noqa: E402

AS_OF = pd.Timestamp("2026-10-20")


def person(**over):
    base = {"id": "T-1", "age": 60, "weight_kg_12w_ago": 70.0, "weight_kg_now": 70.0, "egfr_now": 80, "egfr_prev": 82,
            "egfr_dates": ["2026-07-28", "2026-10-17"], "cyp2c9": "not tested", "g6pd": "not tested",
            "medicines": [("glimepiride", 2.0, "twice daily", "2023-01-01")], "fasting_plan": None}
    base.update(over)
    return base


def names(p, as_of=AS_OF):
    return {f.name for f in check(p, as_of)}


def test_future_glp1_start_is_not_a_recent_start():
    future = person(medicines=[("glimepiride", 2.0, "twice daily", "2023-01-01"),
                               ("semaglutide", 0.5, "weekly", "2026-11-01")])
    assert "GLP-1 start" not in names(future)
    assert "GLP-1 start" in names(future, pd.Timestamp("2026-11-15"))  # once the date has passed


def test_planned_fast_fires_only_within_two_weeks():
    soon = person(fasting_plan={"label": "One-day fast", "dates": ["2026-10-25"], "window": [6, 18]})
    later = person(fasting_plan={"label": "Month-long daytime fast", "dates": ["2027-02-10"], "window": [5, 19]})
    assert "Planned fast" in names(soon) and "Planned fast" not in names(later)


def test_fast_skips_meals_only_on_its_date():
    p = person(fasting_plan={"label": "One-day fast", "dates": ["2026-10-22"], "window": [6, 18]})
    change = body_change_for(p, AS_OF)
    period = routine_period(routine_for(p), pd.Timestamp("2026-10-21"), 3, change.fasting_hours, change.fasting_dates)
    meals_per_day = period.meals.groupby(period.meals["time"].dt.date).size()
    assert meals_per_day[pd.Timestamp("2026-10-22").date()] == 1  # only the evening meal
    assert meals_per_day[pd.Timestamp("2026-10-21").date()] == 4


def test_untested_genetics_raise_no_safety_flag_and_stay_unknown():
    flags = {f.name: f.kind for f in check(person(age=70))}
    assert "CYP2C9 variant" not in flags and "G6PD deficiency" not in flags
    assert flags["CYP2C9 not tested"] == "missing" and flags["G6PD not tested"] == "missing"
    assert body_change_for(person()).su_exposure == 1.0  # no genotype is assumed


def test_kidney_flag_shows_values_and_dates_and_missing_egfr_is_reported():
    low = [f for f in check(person(egfr_now=55)) if f.name == "Kidney function"][0]
    assert "55" in low.reason and "17 Oct 2026" in low.reason and "82" in low.reason
    assert "eGFR not recorded" in names(person(egfr_now=None))


def test_g6pd_deficiency_is_a_safety_flag():
    assert "G6PD deficiency" in names(person(g6pd="deficient"))
