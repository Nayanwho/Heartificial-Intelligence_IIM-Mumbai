"""Fixed settings of the version-2 evaluation protocol (docs/evaluation_protocol.md).

Every number here was fixed before the version-2 experiments were run. Change one only
by writing a new protocol version; never to improve a result.
"""

from __future__ import annotations

PROTOCOL_VERSION = "2.0 (frozen 2026-10-23)"

GRID_MIN = 15  # CGM grid step, minutes
WARMUP_HOURS = 24  # discarded at the start of every sensor period
CALIBRATION_DAYS = 5  # personalisation interval after warm-up, fixed length
MIN_TEST_DAYS = 1.0  # follow-up needed after calibration for a period to be scored
MIN_CALIBRATION_READINGS = 300
FALLBACK_MEAL_CARBS = 45.0  # grams, when fewer than 3 meals were weighed during calibration
MIN_CALIBRATION_MEALS = 3

LAGS_MIN = (15, 30, 60, 120)
HORIZONS_MIN = (30, 60, 120)
TREND_WINDOW_MIN = 30

LOW_MG_DL = 70.0
EVENT_MIN_POINTS = 2
EVENT_MAX_GAP_MIN = 20.0

ALERT_HORIZON_MIN = 30  # forecasts used as alert scores look 30 minutes ahead
ALERT_COOLDOWN_MIN = 60.0  # measured from the last EMITTED alert
WARNING_WINDOW_MIN = 60.0  # 0 < event start - alert time <= 60 minutes
FOLLOW_UP_MAX_GAP_MIN = 20.0
ALERT_BUDGETS_PER_DAY = (0.5, 1.0)  # primary, secondary
PRIMARY_BUDGET = 0.5

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 4
SEED = 7
BOOTSTRAP_REPS = 1000

LGB_PARAMS = dict(learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, n_estimators=400, random_state=SEED, verbose=-1)

SENSOR_OFF_MIN_FINGERSTICKS = 3
SENSOR_OFF_MIN_DAYS = 1.5
