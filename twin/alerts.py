"""One alert policy and one way to score it, shared by the evaluation and the live replay.

Policy (docs/evaluation_protocol.md, section 8): at each origin, in time order, an alert is
emitted when the score passes its threshold, the current reading is at least 70 mg/dL (no
alerts during an ongoing low), and at least 60 minutes have passed since the last EMITTED
alert. Suppressed positives never reset the cooldown. These are software choices, not
clinical standards.

Scoring (section 9): an alert warns of an event when 0 < event start - alert time <= 60 min.
Eligible events are matched one-to-one, each to the earliest unmatched alert in its window.
Other alerts are redundant (an event did follow, but it was already credited), false (no event
followed and the hour after was fully observed) or censored (the hour after was not observed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .protocol import ALERT_COOLDOWN_MIN, FOLLOW_UP_MAX_GAP_MIN, LOW_MG_DL, WARNING_WINDOW_MIN


@dataclass(frozen=True)
class AlertPolicy:
    threshold: float
    kind: str = "glucose"  # "glucose": fire when score <= threshold; "probability": when score >= threshold
    cooldown_min: float = ALERT_COOLDOWN_MIN
    low_mg_dl: float = LOW_MG_DL

    def condition(self, score: np.ndarray, current: np.ndarray) -> np.ndarray:
        score = np.asarray(score, float)
        current = np.asarray(current, float)
        passes = score <= self.threshold if self.kind == "glucose" else score >= self.threshold
        return np.isfinite(score) & np.isfinite(current) & (current >= self.low_mg_dl) & passes


def emit(times, score, current, policy: AlertPolicy) -> np.ndarray:
    """Boolean array: was an alert emitted at each origin? `times` must be one patient-period,
    sorted. The state (time of the last emitted alert) starts empty."""
    t = pd.DatetimeIndex(times)
    minutes = (t - t[0]).total_seconds().to_numpy() / 60.0 if len(t) else np.array([])
    cond = policy.condition(score, current)
    out = np.zeros(len(t), dtype=bool)
    last = -np.inf
    for k in np.flatnonzero(cond):
        if minutes[k] - last >= policy.cooldown_min:
            out[k] = True
            last = minutes[k]
    return out


def alert_rate_per_day(frame: pd.DataFrame, score_col: str, policy: AlertPolicy, grid_min: float = 15.0) -> float:
    """Emitted alerts per monitored day over a set of patient-periods (no outcomes used)."""
    n_alerts, n_origins = 0, 0
    for _, g in frame.groupby("period", sort=False):
        n_alerts += int(emit(g["time"], g[score_col], g["g0"], policy).sum())
        n_origins += len(g)
    days = n_origins * grid_min / 1440.0
    return n_alerts / days if days else np.nan


def choose_threshold(frame: pd.DataFrame, score_col: str, kind: str, budget_per_day: float,
                     candidates: np.ndarray) -> float:
    """The most permissive threshold whose emitted-alert rate does not exceed the budget."""
    order = np.sort(candidates)[::-1] if kind == "glucose" else np.sort(candidates)
    best = None
    # Walk from strict to permissive; keep the last threshold within budget.
    for thr in order[::-1]:
        rate = alert_rate_per_day(frame, score_col, AlertPolicy(float(thr), kind))
        if rate <= budget_per_day:
            best = float(thr)
        else:
            break
    if best is None:
        best = float(order[-1])  # strictest candidate
    return best


def follow_up_observed(alert_time: pd.Timestamp, reading_times: pd.DatetimeIndex,
                       window_min: float = WARNING_WINDOW_MIN, max_gap_min: float = FOLLOW_UP_MAX_GAP_MIN) -> bool:
    """True when readings cover the hour after the alert with no gap over 20 minutes."""
    end = alert_time + pd.Timedelta(minutes=window_min)
    inside = reading_times[(reading_times > alert_time) & (reading_times <= end)]
    seq = pd.DatetimeIndex([alert_time]).append(inside)
    if len(seq) < 2:
        return False
    gaps = np.diff(seq.values).astype("timedelta64[s]").astype(float) / 60.0
    tail = (end - seq[-1]).total_seconds() / 60.0
    return bool(gaps.max() <= max_gap_min and tail < max_gap_min)


def score_alerts(alert_times, event_starts, eligible, reading_times,
                 window_min: float = WARNING_WINDOW_MIN) -> tuple[list[dict], list[dict]]:
    """Match alerts and events of one patient-period. Returns (event records, alert records)."""
    alerts = sorted(pd.DatetimeIndex(alert_times))
    events = list(pd.DatetimeIndex(event_starts))
    order = np.argsort(np.asarray(events, dtype="datetime64[ns]")) if events else []
    events = [events[i] for i in order]
    eligible = [bool(np.asarray(eligible)[i]) for i in order] if len(events) else []
    reading_times = pd.DatetimeIndex(reading_times)
    w = pd.Timedelta(minutes=window_min)
    used: set[int] = set()
    ev_records = []
    for e, ok in zip(events, eligible):
        if not ok:
            continue
        cands = [k for k, a in enumerate(alerts) if k not in used and pd.Timedelta(0) < e - a <= w]
        rec = {"event_start": e, "caught": False, "lead_min": np.nan}
        if cands:
            k = min(cands, key=lambda j: alerts[j])
            used.add(k)
            rec.update(caught=True, lead_min=(e - alerts[k]).total_seconds() / 60.0)
        ev_records.append(rec)
    al_records = []
    for k, a in enumerate(alerts):
        if k in used:
            cls = "matched"
        elif any(pd.Timedelta(0) < e - a <= w for e in events):
            cls = "redundant"
        elif follow_up_observed(a, reading_times, window_min):
            cls = "false"
        else:
            cls = "censored"
        al_records.append({"alert_time": a, "class": cls})
    return ev_records, al_records
