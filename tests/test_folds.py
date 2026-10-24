"""Patient folds keep a person's periods together, and inner threshold selection is isolated like the outer test."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from helpers import synthetic_period

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate  # noqa: E402
from twin.calibration import calibrate, eligibility  # noqa: E402

RESULTS = ROOT / "results"


@pytest.mark.skipif(not (RESULTS / "calibrations.csv").exists(), reason="run scripts/evaluate.py first")
def test_every_period_of_a_patient_is_in_that_patients_fold():
    folds = json.loads((RESULTS / "folds.json").read_text())["patient_fold"]
    cals = pd.read_csv(RESULTS / "calibrations.csv", dtype={"patient": str})
    assert (cals["fold"] == cals["patient"].map(folds)).all()
    repeated = cals.groupby("patient")["fold"].nunique()
    assert (repeated == 1).all() and (cals.groupby("patient").size() > 1).any()
    assert evaluate.assign_folds(sorted(folds)) == folds


def test_inner_plan_partitions_the_training_patients():
    pats = [f"{9000 + i}" for i in range(23)]
    plan = evaluate.inner_plan(pats)
    held_out = [v for _, v in plan]
    for fit, val in plan:
        assert not fit & val and fit | val == set(pats)
    assert set().union(*held_out) == set(pats) and sum(len(v) for v in held_out) == len(pats)


def test_inner_selection_recomputes_the_offset_prior_without_its_held_out_patients(monkeypatch):
    periods = [synthetic_period(seed=i, period_id=f"{9001 + i}_0_20210101") for i in range(8)]
    rows = []
    for p in periods:
        cal = calibrate(p, 5.0)
        assert eligibility(p, cal)[0]
        rows.append(evaluate.period_rows(cal, p)[0])
    train = pd.concat(rows, ignore_index=True)
    seen = []
    real = evaluate.calibration_prior
    monkeypatch.setattr(evaluate, "calibration_prior", lambda ps, pats: seen.append(set(pats)) or real(ps, pats))
    scores = evaluate.inner_learned_scores(periods, train, "corrected")
    plan = evaluate.inner_plan(sorted(train["patient"].unique()))
    assert seen == [fit for fit, _ in plan]
    for (fit, val), prior_patients in zip(plan, seen):
        assert not prior_patients & val
    assert set(scores["patient"]) == set(train["patient"]) and scores[["B_30", "C_30", "p_low"]].notna().all().all()
