"""The live replay uses the evaluation's own feature and model code, and only past data."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from twin.data_shanghai import DATA_DIR, load_period, load_summary  # noqa: E402
from twin.replay import FoldModels, clinical_snapshot, replay_step  # noqa: E402
from twin.triggers import check  # noqa: E402

RESULTS = ROOT / "results"
needs_results = pytest.mark.skipif(not (RESULTS / "predictions.csv.gz").exists() or not (RESULTS / "models").exists(),
                                   reason="run scripts/evaluate.py first")


def setup(period_index=0, fraction=0.5):
    preds = pd.read_csv(RESULTS / "predictions.csv.gz", parse_dates=["time"])
    cals = pd.read_csv(RESULTS / "calibrations.csv").set_index("period")
    ev = json.loads((RESULTS / "evaluation.json").read_text())
    pid = sorted(preds["period"].unique())[period_index]
    rows = preds[preds["period"] == pid].sort_values("time")
    origin = rows.iloc[int(len(rows) * fraction)]
    summary = load_summary().set_index("period_id")
    path = next((DATA_DIR / "periods").glob(f"{pid}.xls*"))
    period = load_period(path, summary.loc[pid].to_dict())
    fold = int(cals.loc[pid, "fold"])
    models = FoldModels.load(fold, float(ev["thresholds_by_fold"][str(fold)]["current"]["budget_0.5"]))
    return period, float(cals.loc[pid, "prior"]), models, origin


@needs_results
def test_replay_reproduces_the_stored_evaluation_prediction():
    period, prior, models, origin = setup()
    out = replay_step(period, prior, models, origin["time"])
    assert out["status"] == "ok" and out["origin"] == origin["time"]
    for h in (30, 60, 120):
        assert abs(out["forecast"][h] - origin[f"C_{h}"]) < 1e-6
    assert abs(out["p_low"] - origin["p_low"]) < 1e-6


@needs_results
def test_replay_output_ignores_records_after_the_clock():
    period, prior, models, origin = setup(1, 0.3)
    base = replay_step(period, prior, models, origin["time"])
    q = copy.deepcopy(period)
    q.cgm = q.cgm.copy()
    q.cgm[q.cgm.index > origin["time"]] = 41.0
    q.fingerstick = q.fingerstick.copy()
    q.fingerstick[q.fingerstick.index > origin["time"]] = 350.0
    other = replay_step(q, prior, models, origin["time"])
    assert other["forecast"] == base["forecast"] and other["alerts"] == base["alerts"]


@needs_results
def test_replay_has_no_hidden_state_between_patients():
    a = setup(0)
    b = setup(2)
    first = replay_step(*a[:3], a[3]["time"])
    replay_step(*b[:3], b[3]["time"])
    again = replay_step(*a[:3], a[3]["time"])
    assert first["forecast"] == again["forecast"] and first["alerts"] == again["alerts"]


@needs_results
def test_replay_reports_insufficient_data_instead_of_guessing():
    period, prior, models, origin = setup()
    early = replay_step(period, prior, models, period.cgm.index.min() + pd.Timedelta(days=2))
    assert early["status"] == "calibrating" and "forecast" not in early
    dropped = replay_step(period, prior, models, origin["time"], dropout_minutes=45)
    assert dropped["status"] == "unavailable" and "forecast" not in dropped


def test_synthetic_context_update_changes_only_the_review_trigger():
    summary = load_summary().set_index("period_id")
    pid = "2045_0_20201216"
    period = load_period(next((DATA_DIR / "periods").glob(f"{pid}.xls*")), summary.loc[pid].to_dict())
    clock = period.cgm.index.max()
    without = {f.name: f for f in check(clinical_snapshot(period, clock), clock)}
    with_glp1 = {f.name: f for f in check(clinical_snapshot(period, clock, synthetic_glp1_weeks=6), clock)}
    assert "GLP-1 start" not in without and "GLP-1 start" in with_glp1
    assert with_glp1["GLP-1 start"].field == "medicines"
    assert np.isfinite(period.cgm.iloc[-1])  # the CGM history itself is untouched


@needs_results
def test_replay_alerts_match_the_evaluation_alert_log():
    """The replay's alerts, emitted by the shared policy, are the evaluation's logged alerts up to the clock."""
    log = pd.read_csv(RESULTS / "alerts.csv.gz", parse_dates=["alert_time"])
    log = log[(log["method"] == "current") & (log["setting"] == "budget_0.5") & log["alert_time"].notna()]
    for index, fraction in ((0, 0.6), (2, 0.9)):
        period, prior, models, origin = setup(index, fraction)
        out = replay_step(period, prior, models, origin["time"])
        logged = log[(log["period"] == period.period_id) & (log["alert_time"] <= origin["time"])]["alert_time"]
        assert sorted(pd.DatetimeIndex(out["alerts"])) == sorted(pd.DatetimeIndex(logged))


@needs_results
def test_replay_matches_stored_predictions_at_many_origins():
    for index in (0, 3, 7, 12):
        for fraction in (0.1, 0.45, 0.9):
            period, prior, models, origin = setup(index, fraction)
            out = replay_step(period, prior, models, origin["time"])
            assert out["status"] == "ok" and out["origin"] == origin["time"]
            for h in (30, 60, 120):
                assert abs(out["forecast"][h] - origin[f"C_{h}"]) < 1e-6
