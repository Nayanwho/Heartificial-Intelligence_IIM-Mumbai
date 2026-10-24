import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twin.sensor import check_sensor, low_events, pair_fingersticks, person_offset, trim_warmup  # noqa: E402


def series(values, start="2021-01-01 00:00", freq="15min"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq), dtype=float)


def test_a_single_low_reading_is_not_an_event():
    s = series([100, 65, 100, 100])
    assert low_events(s).empty


def test_two_readings_in_a_row_make_one_event_with_its_nadir():
    s = series([100, 66, 60, 90, 100])
    ev = low_events(s)
    assert len(ev) == 1 and ev.iloc[0]["nadir"] == 60 and ev.iloc[0]["points"] == 2
    assert bool(ev.iloc[0]["night"]) is True  # starts at 00:15


def test_a_gap_in_the_data_splits_events():
    idx = pd.to_datetime(["2021-01-01 10:00", "2021-01-01 10:15", "2021-01-01 11:00", "2021-01-01 11:15"])
    s = pd.Series([65, 64, 63, 62], index=idx, dtype=float)
    assert len(low_events(s)) == 2


def test_warmup_day_is_dropped():
    s = series(np.full(200, 120.0))
    trimmed = trim_warmup(s, hours=24)
    assert trimmed.index.min() == s.index.min() + pd.Timedelta(hours=24)


def test_offset_is_shrunk_towards_the_prior_when_pairs_are_few():
    pairs = pd.DataFrame({"diff": [20.0, 20.0]})
    assert person_offset(pairs, prior=5.0, k=10) == (2 * 20 + 10 * 5) / 12
    assert person_offset(pairs.iloc[:0], prior=5.0) == 5.0


def test_pairs_only_within_tolerance():
    cgm = series([100, 110, 120])  # 00:00, 00:15, 00:30
    fs = pd.Series([112.0, 150.0], index=pd.to_datetime(["2021-01-01 00:16", "2021-01-01 02:00"]))
    pairs = pair_fingersticks(cgm, fs, tolerance_min=8)
    assert len(pairs) == 1 and pairs.iloc[0]["diff"] == 2.0


def test_correction_lifts_a_sensor_that_reads_low():
    cgm = series(np.r_[np.full(100, 120.0), [65, 62, 66], np.full(20, 120.0)])
    fs_times = cgm.index[100:123:2]
    fs = pd.Series(cgm.loc[fs_times].to_numpy() + 15.0, index=fs_times)
    chk = check_sensor(cgm, fs, prior=0.0)
    assert len(low_events(chk.raw)) == 1
    assert chk.offset > 5 and low_events(chk.corrected).empty
