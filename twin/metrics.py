"""Accuracy measures for glucose forecasts and low alarms."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(pred, truth) -> float:
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def mard(pred, truth) -> float:
    """Mean absolute relative difference, percent."""
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    return float(np.mean(np.abs(pred - truth) / truth) * 100)


def clarke_zone(ref: float, pred: float) -> str:
    """Clarke error grid zone (A to E) for one reference and one predicted value, in mg/dL."""
    if (ref <= 70 and pred <= 70) or (0.8 * ref <= pred <= 1.2 * ref):
        return "A"
    if (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
        return "E"
    if (70 <= ref <= 290 and pred >= ref + 110) or (130 <= ref <= 180 and pred <= (7 / 5) * ref - 182):
        return "C"
    if (ref >= 240 and 70 <= pred <= 180) or (ref <= 175 / 3 and 70 <= pred <= 180) or \
            (175 / 3 <= ref <= 70 and pred >= (6 / 5) * ref):
        return "D"
    return "B"


def clarke_zones(ref, pred) -> np.ndarray:
    """Vectorised `clarke_zone` (same rules, same precedence A, E, C, D, else B)."""
    ref, pred = np.asarray(ref, float), np.asarray(pred, float)
    zone = np.full(len(ref), "B", dtype="<U1")
    d = ((ref >= 240) & (pred >= 70) & (pred <= 180)) | ((ref <= 175 / 3) & (pred >= 70) & (pred <= 180)) | \
        ((ref >= 175 / 3) & (ref <= 70) & (pred >= (6 / 5) * ref))
    c = ((ref >= 70) & (ref <= 290) & (pred >= ref + 110)) | ((ref >= 130) & (ref <= 180) & (pred <= (7 / 5) * ref - 182))
    e = ((ref >= 180) & (pred <= 70)) | ((ref <= 70) & (pred >= 180))
    a = ((ref <= 70) & (pred <= 70)) | ((pred >= 0.8 * ref) & (pred <= 1.2 * ref))
    zone[d] = "D"
    zone[c] = "C"
    zone[e] = "E"
    zone[a] = "A"
    return zone


def clarke_ab(pred, truth) -> float:
    """Share of points in Clarke zones A or B, percent."""
    zones = [clarke_zone(r, p) for r, p in zip(truth, pred)]
    return float(np.mean([z in ("A", "B") for z in zones]) * 100)


def bootstrap_ci(df: pd.DataFrame, stat, group: str = "patient", n: int = 1000, seed: int = 7) -> tuple[float, float]:
    """95% interval for `stat(df)` by resampling whole patients."""
    rng = np.random.default_rng(seed)
    groups = df[group].unique()
    by_group = {g: d for g, d in df.groupby(group)}
    values = []
    for _ in range(n):
        pick = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([by_group[g] for g in pick], ignore_index=True)
        try:
            values.append(stat(sample))
        except (ValueError, ZeroDivisionError):
            continue
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def alarm_episodes(times: pd.Series, alarm: np.ndarray, merge_min: float = 60.0) -> list[pd.Timestamp]:
    """Collapse alarms closer than `merge_min` minutes into single episodes; returns episode starts."""
    starts, last = [], None
    for t, a in zip(pd.to_datetime(times), alarm):
        if not a:
            continue
        if last is None or (t - last).total_seconds() / 60 > merge_min:
            starts.append(t)
        last = t
    return starts
