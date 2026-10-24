"""How much do the illustrative dose scenarios depend on their assumptions?

For the demo persona (Meena), every dose option is simulated under:
- three process-noise multipliers (0.25, 0.5, 1.0 x the core's fitting error), with the
  default body-change assumptions;
- one-at-a-time changes to each body-change assumption (appetite on a GLP-1, extra drug effect
  per weight lost, longer sulfonylurea action in kidney impairment), at noise 0.5.

Outputs results/scenario_sensitivity.json. Run after make_cohort.py:
    python scripts/scenario_sensitivity.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from twin.synthetic import body_change_for, dose_options, twin_from_row  # noqa: E402
from twin.whatif import NOISE_FACTORS, BodyChange, Scenario, compare  # noqa: E402

N_SIMS, DAYS = 300, 14


def table(tw, options, change, noise):
    t = compare(tw, options, change, Scenario(), days=DAYS, n_sims=N_SIMS, noise_factor=noise)
    return {r["option"]: {"simulated_nights_with_low_percent": r["night_low_risk_percent"],
                          "time_in_range_percent": r["time_in_range_percent"],
                          "time_above_180_percent": r["time_above_180_percent"],
                          "mean_glucose_mg_dl": r["mean_glucose_mg_dl"]} for _, r in t.iterrows()}


def main():
    twins = pd.read_csv(ROOT / "results" / "twins.csv")
    people = json.loads((ROOT / "data" / "synthetic" / "people.json").read_text())
    meena = people[0]
    tw = twin_from_row(twins[twins["period"] == meena["seed_period"]].iloc[0])
    options = dose_options(meena)
    base = body_change_for(meena)
    out = {"persona": meena["id"], "seed_period": meena["seed_period"], "days": DAYS, "n_sims": N_SIMS,
           "default_assumptions": {"appetite_cut": base.appetite_cut, "sensitivity_gain": base.sensitivity_gain,
                                   "su_action_factor": base.su_half_life, "su_exposure": base.su_exposure,
                                   "fasting_dates": list(base.fasting_dates)},
           "note": "Simulated, comparative and uncalibrated. Not treatment-effect estimates.",
           "before_changes": table(tw, {"Keep current doses": options["Keep current doses"]},
                                   BodyChange(su_exposure=base.su_exposure), 0.5),
           "by_noise_factor": {str(nf): table(tw, options, base, nf) for nf in NOISE_FACTORS},
           "one_at_a_time": {}}
    variants = {
        "appetite_cut": [0.15, 0.30, 0.45],
        "sensitivity_gain": [round(base.sensitivity_gain * f, 3) for f in (0.5, 1.0, 1.5)],
        "su_half_life": [1.0, 1.2, 1.4, 1.8],
    }
    for field, values in variants.items():
        out["one_at_a_time"][field] = {str(v): table(tw, options, replace(base, **{field: v}), 0.5) for v in values}
    out["without_planned_fast"] = table(tw, options, replace(base, fasting_dates=(), fasting_hours=None), 0.5)
    (ROOT / "results" / "scenario_sensitivity.json").write_text(json.dumps(out, indent=1))
    for nf, t in out["by_noise_factor"].items():
        print(nf, {k: v["simulated_nights_with_low_percent"] for k, v in t.items()})


if __name__ == "__main__":
    main()
