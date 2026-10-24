"""The shared alert policy and its one-to-one scoring."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twin.alerts import AlertPolicy, emit, follow_up_observed, score_alerts  # noqa: E402
from twin.metrics import clarke_zone, clarke_zones  # noqa: E402

T0 = pd.Timestamp("2021-01-01 00:00")


def at(minutes):
    return T0 + pd.Timedelta(minutes=minutes)


def grid(n):
    return pd.DatetimeIndex([at(15 * k) for k in range(n)])


def test_cooldown_counts_from_the_last_emitted_alert_only():
    times = grid(10)
    score = np.full(10, 60.0)  # every origin is positive
    fired = emit(times, score, np.full(10, 100.0), AlertPolicy(80))
    assert list(np.flatnonzero(fired)) == [0, 4, 8]  # every 60 minutes, suppressed positives do not reset it


def test_no_alert_during_an_ongoing_low():
    times = grid(4)
    fired = emit(times, np.full(4, 60.0), np.array([65.0, 64.0, 90.0, 90.0]), AlertPolicy(80))
    assert list(np.flatnonzero(fired)) == [2]


def test_probability_scores_fire_above_threshold():
    fired = emit(grid(3), np.array([0.1, 0.6, 0.2]), np.full(3, 100.0), AlertPolicy(0.5, "probability"))
    assert list(np.flatnonzero(fired)) == [1]


def test_matching_is_one_to_one_and_needs_positive_lead():
    readings = grid(40)
    alerts = [at(60), at(90), at(130)]  # two alerts before the event at 120; one after it starts
    ev, al = score_alerts(alerts, [at(120)], [True], readings)
    assert ev[0]["caught"] and ev[0]["lead_min"] == 60.0  # earliest alert in the window
    assert [a["class"] for a in al] == ["matched", "redundant", "false"]


def test_alert_after_event_start_is_not_credited():
    ev, al = score_alerts([at(130)], [at(120)], [True], grid(40))
    assert not ev[0]["caught"] and al[0]["class"] == "false"


def test_one_alert_cannot_catch_two_events():
    ev, al = score_alerts([at(100)], [at(120), at(150)], [True, True], grid(40))
    assert [e["caught"] for e in ev] == [True, False]
    assert [a["class"] for a in al] == ["matched"]


def test_ineligible_event_is_not_counted_but_its_alert_is_not_false():
    ev, al = score_alerts([at(20)], [at(45)], [False], grid(40))
    assert ev == [] and al[0]["class"] == "redundant"


def test_follow_up_censoring_at_the_end_and_across_gaps():
    readings = grid(10)  # ends at 135 minutes
    assert follow_up_observed(at(60), readings)
    assert not follow_up_observed(at(120), readings)  # record ends inside the hour
    gappy = readings.delete([5, 6])  # a 45-minute gap after 60 minutes
    assert not follow_up_observed(at(60), gappy)
    _, al = score_alerts([at(120)], [], [], readings)
    assert al[0]["class"] == "censored"


def test_vector_clarke_matches_scalar():
    rng = np.random.default_rng(0)
    ref, pred = rng.uniform(30, 400, 3000), rng.uniform(30, 400, 3000)
    assert list(clarke_zones(ref, pred)) == [clarke_zone(r, p) for r, p in zip(ref, pred)]
