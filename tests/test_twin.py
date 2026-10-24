import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twin.model import COLUMNS, Twin, fit_twin, simulate_array  # noqa: E402
from twin.whatif import BodyChange, Routine, compare  # noqa: E402

TRUE = {"const": 9.0, "glucose": -0.06, "insulin": -1.5, "su": -4.0, "glp1": 0.0, "carbs": 0.4,
        "sin24": 1.0, "cos24": -0.5}


def synthetic_data(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2021-01-01", periods=n, freq="15min")
    hours = times.hour + times.minute / 60
    inputs = pd.DataFrame({"insulin": rng.gamma(2, 1, n), "su": rng.gamma(2, 0.5, n), "glp1": 0.0,
                           "carbs": rng.gamma(1, 10, n), "sin24": np.sin(2 * np.pi * hours / 24),
                           "cos24": np.cos(2 * np.pi * hours / 24)}, index=times)
    twin = Twin(coef=TRUE, n_steps=0, residual_sd=0.0)
    g = simulate_array(twin, 150.0, inputs[COLUMNS[2:]].to_numpy(), np.full(n, 15.0))
    return pd.Series(g, index=times), inputs


def test_fit_recovers_known_coefficients():
    g, inputs = synthetic_data()
    tw = fit_twin(g, inputs)
    for k in ("glucose", "insulin", "su", "carbs"):
        assert abs(tw.coef[k] - TRUE[k]) < 0.15 * abs(TRUE[k]) + 0.02, (k, tw.coef[k])


def test_signs_are_constrained():
    g, inputs = synthetic_data()
    tw = fit_twin(g + np.random.default_rng(1).normal(0, 15, len(g)), inputs)
    assert tw.coef["insulin"] <= 0 and tw.coef["su"] <= 0 and tw.coef["carbs"] >= 0 and tw.coef["glucose"] < 0


def test_lower_sulfonylurea_dose_lowers_night_risk():
    twin = Twin(coef={"const": 8.0, "glucose": -0.07, "insulin": 0.0, "su": -12.0, "glp1": 0.0, "carbs": 0.3,
                      "sin24": 0.0, "cos24": 0.0}, n_steps=0, residual_sd=8.0)
    meals = [(8.5, 50), (13.5, 70), (20.5, 60)]
    options = {"keep": Routine(meals=meals, tablets=[(8.0, "glimepiride", 2), (20.0, "glimepiride", 2)]),
               "halve evening": Routine(meals=meals, tablets=[(8.0, "glimepiride", 2), (20.0, "glimepiride", 1)])}
    table = compare(twin, options, BodyChange(su_half_life=1.4), days=4, n_sims=100).set_index("option")
    assert table.loc["halve evening", "night_low_risk_percent"] <= table.loc["keep", "night_low_risk_percent"]
    assert table.loc["halve evening", "mean_glucose_mg_dl"] > table.loc["keep", "mean_glucose_mg_dl"]
