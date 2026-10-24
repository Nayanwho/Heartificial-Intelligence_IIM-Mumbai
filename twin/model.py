"""The physiology core: a small glucose model fitted to one person.

Every 15 minutes, glucose moves by

    change = k * (basal - glucose) - insulin effect - sulfonylurea effect
             - GLP-1 effect + meal effect + time-of-day rhythm

This is a linearised form of the classic Bergman minimal model, extended with
sulfonylurea exposure, GLP-1 exposure and a daily rhythm. Each coefficient is
fitted per person with sign constraints, so every number keeps its physiological
meaning: how strongly this person's glucose returns to baseline, how much a unit of
insulin or a reference sulfonylurea dose lowers it, and how much a gram of
carbohydrate raises it. Simulating forward with changed inputs is the what-if.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

COLUMNS = ["const", "glucose", "insulin", "su", "glp1", "carbs", "sin24", "cos24"]
LOWER = np.array([-np.inf, -0.3, -np.inf, -np.inf, -np.inf, 0.0, -np.inf, -np.inf])
UPPER = np.array([np.inf, -0.005, 0.0, 0.0, 0.0, np.inf, np.inf, np.inf])
STEP_MIN = 15.0
MAX_GAP_MIN = 20.0
RIDGE = 1.0
GLUCOSE_FLOOR, GLUCOSE_CEILING = 40.0, 400.0


@dataclass
class Twin:
    coef: dict[str, float]
    n_steps: int
    residual_sd: float
    meta: dict = field(default_factory=dict)

    @property
    def basal_glucose(self) -> float:
        return self.coef["const"] / -self.coef["glucose"]

    def describe(self) -> dict[str, float]:
        """Coefficients in plain units, per 15 minutes."""
        return {
            "basal_glucose_mg_dl": round(self.basal_glucose, 1),
            "return_rate_per_15min": round(-self.coef["glucose"], 4),
            "insulin_mg_dl_per_unit_per_h": round(-self.coef["insulin"], 3),
            "sulfonylurea_mg_dl_per_reference_dose": round(-self.coef["su"], 3),
            "carb_mg_dl_per_g_per_h": round(self.coef["carbs"], 3),
            "residual_sd_mg_dl": round(self.residual_sd, 2),
        }

    def step(self, glucose: float, x: pd.Series | dict) -> float:
        c = self.coef
        change = (c["const"] + c["glucose"] * glucose + c["insulin"] * x["insulin"] + c["su"] * x["su"]
                  + c["glp1"] * x["glp1"] + c["carbs"] * x["carbs"] + c["sin24"] * x["sin24"]
                  + c["cos24"] * x["cos24"])
        return float(np.clip(glucose + change, GLUCOSE_FLOOR, GLUCOSE_CEILING))

    def simulate(self, start_glucose: float, inputs: pd.DataFrame,
                 anchors: pd.Series | None = None) -> pd.Series:
        """Glucose path over `inputs`' times from `start_glucose`.

        `anchors` (for example fingersticks) reset the path to the measured value at their time,
        which is how the twin coasts with the sensor off.
        """
        values = np.empty(len(inputs))
        g = float(start_glucose)
        anchor_iter = iter(anchors.sort_index().items()) if anchors is not None and len(anchors) else iter(())
        next_anchor = next(anchor_iter, None)
        times = inputs.index
        arr = {col: inputs[col].to_numpy() for col in COLUMNS[2:]}
        c = self.coef
        for i in range(len(inputs)):
            if i > 0:
                x = {k: v[i - 1] for k, v in arr.items()}
                dt = (times[i] - times[i - 1]).total_seconds() / 60.0
                change = (c["const"] + c["glucose"] * g + c["insulin"] * x["insulin"] + c["su"] * x["su"]
                          + c["glp1"] * x["glp1"] + c["carbs"] * x["carbs"] + c["sin24"] * x["sin24"]
                          + c["cos24"] * x["cos24"])
                g = float(np.clip(g + change * dt / STEP_MIN, GLUCOSE_FLOOR, GLUCOSE_CEILING))
            while next_anchor is not None and next_anchor[0] <= times[i]:
                g = float(next_anchor[1])
                next_anchor = next(anchor_iter, None)
            values[i] = g
        return pd.Series(values, index=times, name="twin")


def simulate_array(twin: Twin, start_glucose: float, inputs: np.ndarray, minutes: np.ndarray) -> np.ndarray:
    """Fast simulation. `inputs` columns: insulin, su, glp1, carbs, sin24, cos24;
    `minutes[i]` is the time from row i-1 to row i. Returns glucose at each row."""
    c = twin.coef
    w = np.array([c["insulin"], c["su"], c["glp1"], c["carbs"], c["sin24"], c["cos24"]])
    drive = inputs @ w + c["const"]
    out = np.empty(len(inputs))
    g = float(start_glucose)
    out[0] = g
    for i in range(1, len(inputs)):
        g = g + (drive[i - 1] + c["glucose"] * g) * minutes[i] / STEP_MIN
        g = min(max(g, GLUCOSE_FLOOR), GLUCOSE_CEILING)
        out[i] = g
    return out


def design(glucose: pd.Series, inputs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """One row per consecutive pair of readings no more than 20 minutes apart."""
    g = glucose.reindex(inputs.index)
    dt = inputs.index.to_series().diff().dt.total_seconds().div(60.0).to_numpy()
    nxt = g.shift(-1).to_numpy()
    dt_next = np.r_[dt[1:], np.nan]
    ok = np.isfinite(nxt) & np.isfinite(g.to_numpy()) & (dt_next <= MAX_GAP_MIN) & (dt_next > 0)
    y = (nxt - g.to_numpy()) * STEP_MIN / np.where(np.isfinite(dt_next), dt_next, STEP_MIN)
    X = np.column_stack([np.ones(len(g)), g.to_numpy()] + [inputs[c].to_numpy() for c in COLUMNS[2:]])
    return X[ok], y[ok]


def fit_twin(glucose: pd.Series, inputs: pd.DataFrame, meta: dict | None = None) -> Twin:
    """Fit one person's coefficients with sign constraints and a light ridge penalty."""
    X, y = design(glucose, inputs)
    if len(y) < 50:
        raise ValueError("too little data to fit a twin")
    scale = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
    scale[0] = 1.0
    Xs = X / scale
    penalty = np.sqrt(RIDGE) * np.eye(Xs.shape[1])
    penalty[0, 0] = 0.0  # no penalty on the intercept
    A = np.vstack([Xs, penalty])
    b = np.r_[y, np.zeros(Xs.shape[1])]
    res = lsq_linear(A, b, bounds=(LOWER * scale, UPPER * scale), method="trf")
    beta = res.x / scale
    resid = y - X @ beta
    return Twin(coef=dict(zip(COLUMNS, map(float, beta))), n_steps=len(y),
                residual_sd=float(np.std(resid)), meta=meta or {})
