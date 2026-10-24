"""Version-2 evaluation must not let later data reach an earlier computation."""

import copy

import numpy as np
import pandas as pd
from helpers import synthetic_period

from twin.calibration import boundaries, calibrate, calibration_prior
from twin.features import eligible_origins, origin_features
from twin.sensor_off import sensor_off_frame

NUMERIC_SKIP = {"period", "patient", "time"}


def features_at(period, prior, t):
    cal = calibrate(period, prior)
    return origin_features(cal, np.array([cal.position(t)])).iloc[0]


def assert_same(a, b):
    for col in a.index:
        if col in NUMERIC_SKIP:
            assert a[col] == b[col], col
        else:
            x, y = float(a[col]), float(b[col])
            assert (np.isnan(x) and np.isnan(y)) or abs(x - y) < 1e-9, (col, x, y)


def an_origin(period, prior, fraction=0.4):
    cal = calibrate(period, prior)
    pos = eligible_origins(cal)
    return cal.grid[pos[int(len(pos) * fraction)]]


def test_future_records_cannot_change_an_origin():
    p = synthetic_period()
    t = an_origin(p, 7.0)
    before = features_at(p, 7.0, t)
    q = copy.deepcopy(p)
    later = q.cgm.index > t
    q.cgm = q.cgm.copy()
    q.cgm[later] = 45.0  # future sensor readings
    q.fingerstick = q.fingerstick.copy()
    q.fingerstick[q.fingerstick.index > t] = 333.0  # future fingersticks
    extra_meal = pd.DataFrame({"time": [t + pd.Timedelta(minutes=5)], "text": "x", "items": [[("noodles", 400.0, "g")]],
                               "logged": True})
    q.meals = pd.concat([q.meals, extra_meal], ignore_index=True)  # a meal logged after t
    extra_dose = pd.DataFrame({"time": [t + pd.Timedelta(minutes=10)], "drug": "glimepiride",
                               "drug_class": "sulfonylurea", "dose_mg": 4.0})
    q.tablets = pd.concat([q.tablets, extra_dose], ignore_index=True)  # a dose after t
    assert_same(before, features_at(q, 7.0, t))


def test_replay_truncation_gives_identical_features():
    p = synthetic_period()
    t = an_origin(p, 7.0, 0.7)
    assert_same(features_at(p, 7.0, t), features_at(p.until(t), 7.0, t))


def test_offset_uses_calibration_pairs_only():
    p = synthetic_period()
    warm_end, cal_end = boundaries(p)
    base = calibrate(p, 7.0)
    q = copy.deepcopy(p)
    late = pd.Series([400.0, 20.0], index=[cal_end + pd.Timedelta(minutes=1), cal_end + pd.Timedelta(hours=30)])
    q.fingerstick = pd.concat([q.fingerstick, late]).sort_index()
    assert calibrate(q, 7.0).offset == base.offset
    # A fingerstick just before the cutoff must not pair with the CGM reading just after it.
    r = copy.deepcopy(p)
    r.fingerstick = r.fingerstick[r.fingerstick.index < warm_end]
    r.fingerstick = pd.concat([r.fingerstick, pd.Series([999.0], index=[cal_end - pd.Timedelta(minutes=1)])])
    r.cgm = r.cgm[~((r.cgm.index > cal_end - pd.Timedelta(minutes=15)) & (r.cgm.index < cal_end))]
    assert calibrate(r, 7.0).n_pairs == 0


def test_meal_fill_and_drug_flags_come_from_calibration_only():
    p = synthetic_period()
    _, cal_end = boundaries(p)
    q = copy.deepcopy(p)
    late_meals = pd.DataFrame({"time": [cal_end + pd.Timedelta(hours=h) for h in (1, 5, 9)], "text": "x",
                               "items": [[("noodles", 900.0, "g")]] * 3, "logged": True})
    q.meals = pd.concat([q.meals, late_meals], ignore_index=True)
    assert calibrate(q, 7.0).meal_fill == calibrate(p, 7.0).meal_fill
    r = copy.deepcopy(p)
    r.tablets = r.tablets[r.tablets["time"] >= cal_end].reset_index(drop=True)  # SU first recorded later
    assert calibrate(r, 7.0).static["on_su"] == 0.0


def test_prior_ignores_patients_outside_the_training_set():
    a, b = synthetic_period(seed=1, period_id="9001_0_20210101"), synthetic_period(seed=2, period_id="9002_0_20210101")
    prior = calibration_prior([a, b], {"9001"})
    b2 = copy.deepcopy(b)
    b2.fingerstick = b2.fingerstick + 80.0  # a held-out patient's data changes
    assert calibration_prior([a, b2], {"9001"}) == prior


def test_missing_history_slots_stay_missing():
    p = synthetic_period()
    t = an_origin(p, 7.0)
    q = copy.deepcopy(p)
    q.cgm = q.cgm.drop(t - pd.Timedelta(minutes=60))
    f = features_at(q, 7.0, t)
    assert np.isnan(f["lag60"]) and np.isfinite(f["lag30"])


def test_sensor_off_never_reads_hidden_cgm():
    p = synthetic_period()
    cal = calibrate(p, 7.0)
    base = sensor_off_frame(cal, p)
    q = copy.deepcopy(p)
    q.cgm = q.cgm.copy()
    q.cgm[q.cgm.index >= cal.cal_end] = 333.0
    other = sensor_off_frame(calibrate(q, 7.0), q)
    cols = [c for c in base.columns if c not in ("truth", "period", "patient", "time")]
    assert np.allclose(base[cols].to_numpy(float), other[cols].to_numpy(float), equal_nan=True)
    assert base["time"].iloc[0] >= cal.cal_end


def test_pump_basal_pulses_up_to_a_clock_do_not_depend_on_later_records():
    from twin.inputs import insulin_events
    p = synthetic_period(days=9.0)
    starts = pd.date_range("2021-01-01 00:10", periods=36, freq="5h")
    p.pump_basal = pd.Series([0.4, 0.8, 0.6, 1.0] * 9, index=starts)
    full = insulin_events(p)
    for clock in (pd.Timestamp("2021-01-03 18:40"), pd.Timestamp("2021-01-05 07:25"), pd.Timestamp("2021-01-07 23:55")):
        cut = insulin_events(p.until(clock))
        a = full[full["time"] <= clock].sort_values(["time", "profile"]).reset_index(drop=True)
        b = cut[cut["time"] <= clock].sort_values(["time", "profile"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
