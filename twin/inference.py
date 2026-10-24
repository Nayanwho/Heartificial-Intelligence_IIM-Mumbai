"""Shared inference: the one place where LightGBM boosters turn origin features into forecasts.

Both the evaluation (`scripts/evaluate.py`, with freshly fitted fold models) and the live replay
(`twin/replay.py`, with the same models loaded from `results/models/`) call these functions, so
the two cannot drift apart. Models predict the change in glucose at each horizon; the forecast is
the current corrected reading plus that change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def booster_of(model):
    """A LightGBM Booster from either a Booster or a fitted scikit-learn wrapper."""
    return model.booster_ if hasattr(model, "booster_") else model


def forecast(model, feats: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Forecast glucose (mg/dL) at one horizon: current corrected reading + predicted change."""
    return feats["g0"].to_numpy(float) + booster_of(model).predict(feats[features])


def low_probability(model, feats: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Probability that a low begins within the warning window (binary booster output)."""
    return booster_of(model).predict(feats[features])
