"""Figures for the README and the deck, drawn only from the machine-readable results.

Run from the repository root after evaluate.py, make_cohort.py and scenario_sensitivity.py:
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from twin.synthetic import body_change_for, dose_options, twin_from_row  # noqa: E402
from twin.whatif import Scenario, rehearse  # noqa: E402

OUT = ROOT / "docs" / "figures"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"
BLUE, ORANGE, AQUA, GRAY, LIGHT_BLUE, LIGHT_GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8d8c86", "#9cc3ef", "#c9c8c2"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

FORECAST = [("persistence", "No change", LIGHT_GRAY), ("trend", "Recent trend", GRAY),
            ("physiology", "Physiology core only", LIGHT_BLUE), ("A", "ML, dynamic only", "#f3a07a"),
            ("B", "ML + clinical record", ORANGE), ("D", "ML + physiology", AQUA),
            ("C", "Hybrid (ML + record + physiology)", BLUE)]
ALERTS = [("current", "Current reading", INK), ("trend", "Trend forecast", GRAY),
          ("physiology", "Physiology forecast", LIGHT_BLUE), ("ml", "ML forecast", ORANGE),
          ("hybrid", "Hybrid forecast", BLUE), ("classifier", "Hybrid classifier", AQUA)]


def title(fig, main, sub):
    """Title and subtitle at the top left; returns the top of the plotting area as a figure fraction.

    Plain text rather than suptitle, so tight_layout does not reserve extra space for it, and the
    plotting area starts a fixed 0.12 inch below the subtitle whatever the figure height.
    """
    h = fig.get_figheight()
    fig.text(0.012, 1 - 0.06 / h, main, ha="left", va="top", fontsize=13.5, fontweight="bold", color=INK)
    t = fig.text(0.012, 1 - 0.36 / h, sub, ha="left", va="top", fontsize=9.6, color=MUTED, linespacing=1.4)
    fig.canvas.draw()
    return t.get_window_extent().transformed(fig.transFigure.inverted()).y0 - 0.12 / h


def sensor_figure(dc):
    lows = dc["lows"]
    fig, ax = plt.subplots(figsize=(8.4, 3.7))
    labels = ["Raw sensor", "After each person's\noffset correction"]
    values = [lows["raw_15min_lows"], lows["corrected_15min_lows"]]
    bars = ax.barh(labels[::-1], values[::-1], color=[BLUE, GRAY], height=0.5)
    for b, v in zip(bars, values[::-1]):
        ax.text(v + 4, b.get_y() + b.get_height() / 2, f"{v}", va="center", color=INK, fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(values) * 1.15)
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_xlabel("CGM low episodes (15 minutes or more below 70 mg/dL), 100 patients, whole periods")
    s = dc["sensor_check"]
    top = title(fig, f"A whole-period offset correction cuts sensor low episodes from {lows['raw_15min_lows']} to {lows['corrected_15min_lows']}",
          f"Descriptive check: each person's offset from all of their fingersticks (the twin itself uses calibration fingersticks only).\n"
          f"The sensor read {abs(s['mean_sensor_minus_fingerstick_mg_dl']):g} mg/dL below fingersticks on average; "
          f"{s['of_which_fingerstick_also_below_70']} of {s['sensor_lows_with_a_fingerstick']} paired sensor readings below 70\n"
          "(fingerstick within 8 minutes) were confirmed below 70. Fingersticks are sparse, so true lows are not established.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "sensor_check.png", dpi=200)
    plt.close(fig)


def forecast_figure(ev):
    f = ev["forecast"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, h in zip(axes, ("30 min", "60 min", "120 min")):
        names = [n for _, n, _ in FORECAST][::-1]
        vals = [f[h][k]["rmse_mg_dl"] for k, _, _ in FORECAST][::-1]
        lo = [f[h][k]["rmse_mg_dl"] - f[h][k]["rmse_95ci"][0] for k, _, _ in FORECAST][::-1]
        hi = [f[h][k]["rmse_95ci"][1] - f[h][k]["rmse_mg_dl"] for k, _, _ in FORECAST][::-1]
        cols = [c for _, _, c in FORECAST][::-1]
        ax.barh(names, vals, color=cols, height=0.62, xerr=[lo, hi], error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
        for y, v in enumerate(vals):
            ax.text(v + max(vals) * 0.04 + hi[y], y, f"{v:.1f}", va="center", fontsize=8.8, color=INK)
        ax.set_title(f"{h} ahead", fontsize=10.5, color=INK, loc="left")
        ax.xaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(vals) * 1.32)
        ax.set_xlabel("RMSE (mg/dL), lower is better")
    c, p = f["120 min"]["C"]["rmse_mg_dl"], f["120 min"]["persistence"]["rmse_mg_dl"]
    top = title(fig, f"Two hours ahead, the hybrid twin's error is {1 - c / p:.0%} below assuming no change",
          f"{f['30 min']['patients']} held-out patients after a 5-day personal calibration; leak-free protocol v2; "
          "whiskers are 95% patient-bootstrap intervals.\nA straight-line trend is worse than no change.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "forecast_accuracy.png", dpi=200)
    plt.close(fig)


def ingredients_figure(ev):
    pdiff = ev["paired_differences"]
    comps = [("D minus A (value of the physiology core, without record)", "Physiology core, added to ML", AQUA),
             ("C minus B (value of the physiology core)", "Physiology core, added to ML + record", BLUE),
             ("B minus A (value of the clinical record, without physiology)", "Clinical record, added to ML", ORANGE),
             ("C minus D (value of the clinical record, with physiology)", "Clinical record, added to ML + physiology", "#f3a07a")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, h in zip(axes, ("30 min", "60 min", "120 min")):
        names = [n for _, n, _ in comps][::-1]
        vals = [pdiff[h][k]["rmse_difference_mg_dl"] for k, _, _ in comps][::-1]
        cis = [pdiff[h][k]["95ci"] for k, _, _ in comps][::-1]
        cols = [c for _, _, c in comps][::-1]
        ax.barh(names, vals, color=cols, height=0.6,
                xerr=[[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]],
                error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(f"{h} ahead", fontsize=10.5, color=INK, loc="left")
        ax.xaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax.set_xlabel("Change in RMSE (mg/dL)", fontsize=9.5)
    top = title(fig, "The physiology core improves forecasts; the clinical record does not",
          "Paired comparisons on identical origins and patients, 95% patient-bootstrap intervals; left of zero means the "
          "ingredient lowers error.\nThe clinical record's role is the review triggers, not forecast accuracy.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "ingredients.png", dpi=200)
    plt.close(fig)


def alarm_figure(ev):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    for ax, (setting, lab) in zip(axes, (("budget_0.5", "0.5 alerts a day"), ("budget_1.0", "1.0 alert a day"))):
        a = ev["alarms"][setting]
        names = [n for _, n, _ in ALERTS][::-1]
        vals = [a[k]["sensitivity_percent"] for k, _, _ in ALERTS][::-1]
        cis = [a[k]["sensitivity_95ci"] for k, _, _ in ALERTS][::-1]
        cols = [c for _, _, c in ALERTS][::-1]
        ax.barh(names, vals, color=cols, height=0.6,
                xerr=[[v - c[0] for v, c in zip(vals, cis)], [c[1] - v for v, c in zip(vals, cis)]],
                error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
        for y, k in enumerate([k for k, _, _ in ALERTS][::-1]):
            ax.text(103, y, f"{a[k]['caught']}/{a[k]['eligible_events']}", va="center", fontsize=8.8, color=INK)
        ax.set_xlim(0, 115)
        ax.set_title(f"Budget: {lab} (set on training patients)", fontsize=10.2, color=INK, loc="left")
        ax.xaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax.set_xlabel("Lows caught before they began (%)")
    n = ev["alarms"]["budget_0.5"]["current"]
    top = title(fig, "At the same alert budget, a simple rule on the current reading caught the most lows",
          f"{n['eligible_events']} CGM-defined low episodes in {n['patients_with_eligible_events']} held-out patients; "
          "one-to-one matching, warning window 60 minutes, 60-minute cooldown. Intervals overlap widely:\n"
          "no model is shown to improve low alerts.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "alarm_comparison.png", dpi=200)
    plt.close(fig)


def sensor_off_figure(ev):
    so = ev["sensor_off"]
    by = pd.DataFrame(so["by_hours_since_anchor"])
    labels = ["0 to 2", "2 to 4", "4 to 8", "8 to 12", "12 to 48", "48+"][:len(by)]
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    for col, lab, color in (("carried_rmse", "Last fingerstick carried forward", GRAY),
                            ("typical_rmse", "Person's typical day (calibration)", LIGHT_BLUE),
                            ("hybrid_rmse", "Learning layer (experimental)", BLUE)):
        ax.plot(labels, by[col], color=color, lw=2, marker="o", ms=6, label=lab)
    ax.set_ylim(0, by["carried_rmse"].max() * 1.4)
    ax.set_xlabel("Hours since the last fingerstick (or the end of calibration)")
    ax.set_ylabel("RMSE (mg/dL)")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=3)
    top = title(fig, "Sensor off: estimates track the level, not the lows",
          f"{so['hours']:,.0f} hours in {so['patients']} patients, at most {so['days_scored_per_period']['max']:.1f} days per "
          f"period. RMSE {so['hybrid']['rmse_mg_dl']:.1f} vs {so['typical']['rmse_mg_dl']:.1f} (typical day) and "
          f"{so['carried']['rmse_mg_dl']:.1f} (last fingerstick);\nonly {so['hybrid']['share_of_readings_below_70_estimated_below_80_percent']:g}% "
          "of readings below 70 were estimated below 80. Acute low alerts are not available in this mode.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "sensor_off.png", dpi=200)
    plt.close(fig)


def scenario_figure(sens):
    opts = list(sens["by_noise_factor"]["0.5"].keys())
    short = {"Keep current doses": "Keep\ndoses", "Halve the evening sulfonylurea": "Halve\nevening SU",
             "Stop the evening sulfonylurea": "Stop\nevening SU", "Halve every sulfonylurea dose": "Halve every\nSU dose"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    x = np.arange(len(opts))
    for i, (nf, color) in enumerate((("0.25", LIGHT_BLUE), ("0.5", BLUE), ("1.0", "#1e5fae"))):
        vals = [sens["by_noise_factor"][nf][o]["simulated_nights_with_low_percent"] for o in opts]
        axes[0].bar(x + (i - 1) * 0.26, vals, width=0.24, color=color, label=f"noise x{nf}")
    axes[0].set_title("Process-noise multiplier (default assumptions)", fontsize=10.2, loc="left", color=INK)
    for i, (v, color) in enumerate((("1.0", LIGHT_GRAY), ("1.2", LIGHT_BLUE), ("1.4", BLUE), ("1.8", "#1e5fae"))):
        vals = [sens["one_at_a_time"]["su_half_life"][v][o]["simulated_nights_with_low_percent"] for o in opts]
        axes[1].bar(x + (i - 1.5) * 0.2, vals, width=0.19, color=color, label=f"x{v}")
    axes[1].set_title("Assumed longer sulfonylurea action in kidney impairment (noise x0.5)", fontsize=10.2, loc="left",
                      color=INK)
    for ax in axes:
        ax.set_xticks(x, [short.get(o, o) for o in opts], fontsize=9)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right")
    axes[0].set_ylabel("Simulated nights with a low (%)")
    top = title(fig, "Meena's scenario: the order holds, but the percentages swing with the assumptions",
          "Synthetic persona; physiology copied from one real patient's fitted core. 300 simulations x 14 nights per option. "
          "Comparative and uncalibrated:\nthese are not risk estimates or treatment effects.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "scenario_sensitivity.png", dpi=200)
    plt.close(fig)


def pct(v):
    """99.0 -> '99%', 7.5 -> '7.5%': one decimal only where it matters, matching the deck and README."""
    return f"{v:.1f}".rstrip("0").rstrip(".") + "%"


def rehearsal_figure():
    twins = pd.read_csv(ROOT / "results" / "twins.csv")
    people = json.loads((ROOT / "data" / "synthetic" / "people.json").read_text())
    meena = people[0]
    tw = twin_from_row(twins[twins["period"] == meena["seed_period"]].iloc[0])
    change = body_change_for(meena)
    colors = [ORANGE, AQUA, BLUE, GRAY, LIGHT_BLUE]
    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    for (name, routine), color in zip(dose_options(meena).items(), colors):
        r = rehearse(tw, routine, Scenario(), change, days=14, n_sims=300)
        t = r["times"]
        night = (t >= t[0] + pd.Timedelta(hours=18)) & (t < t[0] + pd.Timedelta(hours=32))
        ax.plot(t[night], r["path"][night], color=color, lw=2,
                label=f"{name}: {pct(100 * r['night_low_risk'])} of simulated nights")
    ax.axhline(70, color=MUTED, lw=1, ls=":")
    ax.text(t[night][0], 72, "70 mg/dL", color=MUTED, fontsize=9)
    ax.set_ylabel("Simulated glucose (mg/dL)")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    top = title(fig, "Meena's scenario: one simulated evening and night under each option",
          "Illustrative. After a GLP-1 start, 5% weight loss and eGFR 58, with stated assumptions (noise x0.5).\n"
          "Synthetic patient; physiology copied from one real ShanghaiT2DM patient's fitted core. Not a recommendation.")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(OUT / "step_down_rehearsal.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    ev = json.loads((ROOT / "results" / "evaluation.json").read_text())
    dc = json.loads((ROOT / "results" / "data_check.json").read_text())
    sens = json.loads((ROOT / "results" / "scenario_sensitivity.json").read_text())
    sensor_figure(dc)
    forecast_figure(ev)
    ingredients_figure(ev)
    alarm_figure(ev)
    sensor_off_figure(ev)
    scenario_figure(sens)
    rehearsal_figure()
    print("figures in", OUT)
