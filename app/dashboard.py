"""Santulan research prototype: clinician-facing dashboard.

Run from the repository root:  streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from twin.calibration import boundaries  # noqa: E402
from twin.data_shanghai import load_all  # noqa: E402
from twin.replay import FoldModels, clinical_snapshot, replay_step  # noqa: E402
from twin.synthetic import body_change_for, dose_options, routine_for, twin_from_row  # noqa: E402
from twin.triggers import DEMO_AS_OF, check  # noqa: E402
from twin.whatif import NOISE_FACTORS, BodyChange, Scenario, compare, rehearse  # noqa: E402

st.set_page_config(page_title="Santulan", layout="wide", initial_sidebar_state="collapsed")
INK, ACCENT, WARN, MUTED = "#1f2933", "#2563eb", "#c2410c", "#6b7785"
RESULTS = ROOT / "results"


# ---------------------------------------------------------------- cached loads

@st.cache_data
def load_results():
    ev = json.loads((RESULTS / "evaluation.json").read_text())
    raw = json.loads((RESULTS / "evaluation_raw.json").read_text())
    sens_path = RESULTS / "scenario_sensitivity.json"
    sens = json.loads(sens_path.read_text()) if sens_path.exists() else None
    cals = pd.read_csv(RESULTS / "calibrations.csv", parse_dates=["warm_end", "cal_end"])
    folds = json.loads((RESULTS / "folds.json").read_text())["patient_fold"]
    events = pd.read_csv(RESULTS / "events.csv", parse_dates=["event_start"])
    return ev, raw, sens, cals, folds, events


@st.cache_data
def load_cohort():
    twins = pd.read_csv(RESULTS / "twins.csv")
    people = json.loads((ROOT / "data" / "synthetic" / "people.json").read_text())
    return twins, people


@st.cache_data
def stored_predictions():
    return pd.read_csv(RESULTS / "predictions.csv.gz", parse_dates=["time"])


@st.cache_resource
def periods_by_id():
    return {p.period_id: p for p in load_all()}


@st.cache_data
def alert_log():
    """The evaluation's own alert log for the rule the replay uses (current reading, 0.5 alerts a day)."""
    al = pd.read_csv(RESULTS / "alerts.csv.gz", parse_dates=["alert_time"])
    return al[(al["method"] == "current") & (al["setting"] == "budget_0.5") & al["alert_time"].notna()]


@st.cache_resource
def fold_models(fold: int, threshold: float):
    return FoldModels.load(fold, threshold)


ev, ev_raw, sens, cals, folds, events = load_results()
twins, people = load_cohort()
by_id = {p["id"]: p for p in people}

st.title("Santulan")
st.caption("Research prototype for clinician-led medication review in Type 2 diabetes. Real-patient results are "
           "retrospective (evaluation protocol v2). Synthetic patients and dose scenarios are illustrative; treatment "
           "effects and simulated risks are not validated. The doctor decides.")
with st.sidebar:
    st.markdown("**Demo date (as of)**")
    as_of = pd.Timestamp(st.date_input("Records are read as of", value=DEMO_AS_OF.date()))
    st.caption("Triggers, medication starts and planned fasts are judged as of this date.")
st.caption(f"Synthetic records are read as of **{as_of:%d %b %Y}** (change it in the sidebar). Real-patient replay "
           "dates are the dataset's own (2020 to 2022).")

tab_replay, tab_list, tab_patient, tab_scen, tab_evidence, tab_data = st.tabs(
    ["Live replay (real patient)", "Clinic list (synthetic)", "Patient (synthetic)", "Dose scenarios (illustrative)",
     "Evidence (retrospective)", "Data formats (example shapes)"])


def twin_for(person):
    return twin_from_row(twins[twins["period"] == person["seed_period"]].iloc[0])


def label(pid):
    return pid + (f" ({by_id[pid]['persona']})" if by_id[pid].get("persona") else "")


# ---------------------------------------------------------------- live replay

with tab_replay:
    st.subheader("Live replay on a real, held-out patient (ShanghaiT2DM)")
    st.caption("The replay clock moves through a real sensor period. Only records up to the clock are used. Forecasts "
               "come from the population model of the fold that excluded this patient, through the same feature and "
               "model code as the evaluation. The low alert is the simple current-reading rule (it did at least as well "
               "as the models in evaluation), with the threshold chosen on that fold's training patients.")
    eligible = cals[cals["eligible"]].copy()
    ev_counts = events[events["eligible"]].groupby("period").size()
    eligible["events"] = eligible["period"].map(ev_counts).fillna(0).astype(int)
    eligible = eligible.sort_values(["events", "period"], ascending=[False, True])
    pid = st.selectbox("Sensor period", eligible["period"].tolist(), index=0,
                       format_func=lambda x: f"{x} ({int(eligible.set_index('period').loc[x, 'events'])} CGM low episodes after calibration)")
    row = eligible.set_index("period").loc[pid]
    period = periods_by_id()[pid]
    fold = int(row["fold"])
    thr = float(ev["thresholds_by_fold"][str(fold)]["current"]["budget_0.5"])
    models = fold_models(fold, thr)
    warm_end, cal_end = boundaries(period)
    last = period.cgm.index.max()
    key = f"clock_{pid}"
    if key not in st.session_state:
        first_ev = events[(events["period"] == pid) & events["eligible"]]["event_start"].min()
        st.session_state[key] = (first_ev - pd.Timedelta(minutes=75)) if pd.notna(first_ev) else cal_end + pd.Timedelta(hours=6)
    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("Reset to calibration end", key=f"reset_{pid}"):
        st.session_state[key] = cal_end
    if c2.button("+15 min", key=f"s15_{pid}"):
        st.session_state[key] = min(st.session_state[key] + pd.Timedelta(minutes=15), last)
    if c3.button("+1 hour", key=f"s60_{pid}"):
        st.session_state[key] = min(st.session_state[key] + pd.Timedelta(hours=1), last)
    if c4.button("Back 1 hour", key=f"b60_{pid}"):
        st.session_state[key] = max(st.session_state[key] - pd.Timedelta(hours=1), period.cgm.index.min())
    if c5.button("Show calibration phase", key=f"cal_{pid}"):
        st.session_state[key] = warm_end + pd.Timedelta(days=2)
    evs_p = events[(events["period"] == pid) & events["eligible"]].sort_values("event_start")
    jump_none = "(choose a low episode)"
    jump_labels = {str(t): f"{t:%d %b %H:%M}" + (" (night)" if n else "") for t, n in zip(evs_p["event_start"], evs_p["night"])}

    def _jump(jkey=f"jump_{pid}", ckey=key):
        choice = st.session_state[jkey]
        if choice != jump_none:
            st.session_state[ckey] = pd.Timestamp(choice) - pd.Timedelta(minutes=75)

    st.selectbox("Jump to 75 minutes before a CGM low episode in this period (caught or missed)",
                 [jump_none] + list(jump_labels), key=f"jump_{pid}", on_change=_jump,
                 format_func=lambda x: jump_labels.get(x, x))
    clock = pd.Timestamp(st.session_state[key])
    o1, o2, o3 = st.columns(3)
    dropout = o1.checkbox("Simulate a sensor dropout (hide the last 45 minutes of CGM)", value=False)
    reveal = o2.checkbox("Reveal what happened next (scoring only, never used by the model)", value=False)
    glp1 = o3.checkbox("Synthetic context update: GLP-1 started 6 weeks ago", value=False)
    out = replay_step(period, float(row["prior"]), models, clock, dropout_minutes=45 if dropout else 0)

    st.markdown(f"**Replay clock** {clock:%d %b %Y %H:%M} &nbsp;·&nbsp; **Calibration** "
                f"{cal_end - pd.Timedelta(days=5):%d %b %H:%M} to {cal_end:%d %b %H:%M} &nbsp;·&nbsp; **Model** fold {fold} "
                f"(this patient excluded from training) &nbsp;·&nbsp; **Alert rule** corrected reading at or below {thr:.1f} mg/dL")
    if out["status"] == "ok":
        st.success(out["message"])
        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("Last reading (corrected)", f"{out['g0']:.0f} mg/dL", help=f"at {out['origin']:%H:%M}")
        f2.metric("Forecast +30 min", f"{out['forecast'][30]:.0f}")
        f3.metric("Forecast +60 min", f"{out['forecast'][60]:.0f}")
        f4.metric("Forecast +120 min", f"{out['forecast'][120]:.0f}")
        f5.metric("Alert now", "YES" if out["alert_now"] else "no")
        if out["alert_now"]:
            st.warning(f"Low alert: the corrected reading {out['g0']:.0f} mg/dL is at or below {thr:.1f} "
                       "(cooldown 60 minutes after each emitted alert). In the retrospective evaluation this simple rule "
                       "caught at least as many lows as the forecast models at the same alert budget, so it is the "
                       "alert used here; the forecasts are shown for context.")
        elif out["g0"] < 70:
            st.info(f"The corrected reading is {out['g0']:.0f} mg/dL: a low may be in progress. The rule warns only "
                    "before a low begins, so it does not alert while the reading is below 70.")
        if reveal and out.get("alerts"):
            last_alert = max(out["alerts"])
            logged = alert_log()
            logged = logged[(logged["period"] == pid) & (logged["alert_time"] == last_alert)]
            if len(logged):
                cls = logged["class"].iloc[0]
                nxt = evs_p[(evs_p["event_start"] > last_alert) & (evs_p["event_start"] <= last_alert + pd.Timedelta(minutes=60))]
                meaning = {"matched": "matched: a low began within the next hour",
                           "false": "false: no low began within the next hour",
                           "redundant": "redundant: its low had already been warned",
                           "censored": "censored: the next hour was not fully observed"}.get(cls, cls)
                lead = (f" (low began {nxt['event_start'].iloc[0]:%H:%M}, "
                        f"{(nxt['event_start'].iloc[0] - last_alert).total_seconds() / 60:.0f} minutes after the alert)"
                        if cls == "matched" and len(nxt) else "")
                st.caption(f"Scoring, revealed: the evaluation's alert log has this period's latest alert at "
                           f"{last_alert:%d %b %H:%M} too, scored {meaning}{lead}.")
    elif out["status"] == "calibrating":
        st.info(out["message"])
    else:
        st.error(out["message"])

    fig = go.Figure()
    hist = out.get("history")
    if hist is not None and len(hist):
        h = hist[hist.index >= clock - pd.Timedelta(hours=24)]
        fig.add_trace(go.Scatter(x=h.index, y=h.values, name="CGM (corrected by the frozen calibration offset)",
                                 line=dict(color=INK, width=1.6)))
    if out["status"] == "ok":
        xs = [out["origin"]] + [out["origin"] + pd.Timedelta(minutes=m) for m in (30, 60, 120)]
        ys = [out["g0"]] + [out["forecast"][m] for m in (30, 60, 120)]
        fig.add_trace(go.Scatter(x=xs, y=ys, name="Forecast from the last reading", mode="lines+markers",
                                 line=dict(color=ACCENT, width=2, dash="dash")))
    for a in out.get("alerts", []):
        if a >= clock - pd.Timedelta(hours=24):
            fig.add_vline(x=a, line=dict(color=WARN, width=1, dash="dot"))
    if reveal:
        cal = out.get("cal")
        future = period.cgm[(period.cgm.index > clock) & (period.cgm.index <= clock + pd.Timedelta(hours=3))]
        offset = cal.offset if cal is not None else float(row["offset"])
        fig.add_trace(go.Scatter(x=future.index, y=future.values + offset, name="What happened next (revealed)",
                                 line=dict(color=MUTED, width=1.4, dash="dot")))
    fig.add_hline(y=70, line=dict(color=WARN, dash="dot", width=1))
    fig.update_layout(height=380, yaxis_title="Glucose (mg/dL)", margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="white", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    st.caption("Dotted orange lines: alerts emitted by the shared alert policy up to the clock.")

    if out["status"] == "ok":
        stored = stored_predictions()
        match = stored[(stored["period"] == pid) & (stored["time"] == out["origin"])]
        if len(match):
            st.caption(f"Parity check with the evaluation's stored prediction for this origin: live +30 min "
                       f"{out['forecast'][30]:.2f} vs stored {float(match['C_30'].iloc[0]):.2f} mg/dL.")
        prov = out["inputs_logged"]
        st.caption(f"Inputs up to the clock: {prov['meals']} meal records, {prov['insulin_records']} insulin records, "
                   f"{prov['tablet_records']} tablet records, {prov['fingersticks']} fingersticks. Last CGM reading "
                   f"{out['last_reading']:%d %b %H:%M}. Sensor offset {out['cal'].offset:+.1f} mg/dL, frozen at the end "
                   "of calibration. Physiology core fitted once on the calibration interval, not refitted online.")
    snap = clinical_snapshot(period, clock, synthetic_glp1_weeks=6 if glp1 else None)
    flags = [f for f in check(snap, clock) if f.kind in ("trigger", "safety")]
    st.markdown("**Clinician-review rules on this patient's record at the clock**")
    if flags:
        for f in flags:
            st.warning(f"{f.name} (field: {f.field}): {f.reason}. {f.note}")
    else:
        st.write("No review rule fires.")
    if glp1:
        st.caption("The GLP-1 start is a SYNTHETIC context update for this demo, not part of the real record. It changes "
                   "the review trigger only; it does not refit the physiology core.")

# ---------------------------------------------------------------- clinic list

with tab_list:
    st.subheader("Who may need a dose review (synthetic cohort)")


    @st.cache_data(show_spinner=False)
    def night_share(pid: str, as_of_str: str) -> float:
        person = by_id[pid]
        a = pd.Timestamp(as_of_str)
        r = rehearse(twin_for(person), routine_for(person), Scenario(), body_change_for(person, a), days=3, n_sims=150,
                     start=a + pd.Timedelta(days=1))
        return 100 * r["night_low_risk"]


    rows = []
    for p in people:
        flags = check(p, as_of)
        rows.append({"Patient": label(p["id"]), "Age": p["age"],
                     "Review triggers": ", ".join(f.name for f in flags if f.kind == "trigger") or "none",
                     "Safety notes": ", ".join(f.name for f in flags if f.kind == "safety") or "none",
                     "Not recorded": ", ".join(f.name.replace(" not tested", "").replace(" not recorded", "")
                                               for f in flags if f.kind == "missing") or "none",
                     "Simulated nights with a low, next 3 nights (%)": round(night_share(p["id"], str(as_of.date())), 1)})
    table = pd.DataFrame(rows).sort_values("Simulated nights with a low, next 3 nights (%)", ascending=False)
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Simulated share: average over the next 3 simulated nights of at least 15 minutes below 70 mg/dL, on "
               "current doses with this person's assumptions applied. Comparative and uncalibrated; it is not a "
               "validated probability. Personas are synthetic; each persona's physiology coefficients are copied from "
               "one real ShanghaiT2DM patient's calibration-fitted core.")

# ---------------------------------------------------------------- patient

with tab_patient:
    pidp = st.selectbox("Patient", [p["id"] for p in people], index=0, format_func=label)
    person = by_id[pidp]
    c1, c2, c3 = st.columns(3)
    c1.metric("Age", person["age"])
    ed = person.get("egfr_dates") or [None, None]
    c2.metric("eGFR", person["egfr_now"], delta=person["egfr_now"] - person["egfr_prev"],
              help=f"{person['egfr_prev']} on {ed[0]}; {person['egfr_now']} on {ed[1]}")
    c3.metric("Weight (kg)", person["weight_kg_now"], delta=round(person["weight_kg_now"] - person["weight_kg_12w_ago"], 1),
              help=f"{person['weight_kg_12w_ago']} kg on {(person.get('weight_dates') or ['?'])[0]}")
    st.markdown("**Medicines:** " + "; ".join(f"{m[0]} {m[1]:g} {'IU' if m[0].startswith('insulin') else 'mg'} {m[2]} (since {m[3]})"
                                             for m in person["medicines"]))
    plan = person.get("fasting_plan")
    st.markdown(f"**Genetics:** CYP2C9 {person['cyp2c9']}; G6PD {person['g6pd']}. **Planned fast:** "
                + (f"{plan['label']} on {', '.join(plan['dates'][:3])}{' ...' if len(plan['dates']) > 3 else ''}" if plan else "none"))
    for f in check(person, as_of):
        if f.kind == "trigger":
            st.warning(f"**{f.name}:** {f.reason}. {f.note}")
        elif f.kind == "safety":
            st.info(f"**{f.name}:** {f.reason}. {f.note}")
        else:
            st.caption(f"{f.name}: {f.note}")
    change = body_change_for(person, as_of)
    r = rehearse(twin_for(person), routine_for(person), Scenario(), change, days=3, n_sims=150,
                 start=as_of + pd.Timedelta(days=1))
    fig = go.Figure()
    t = r["times"]
    fig.add_trace(go.Scatter(x=list(t) + list(t[::-1]), y=list(r["p90"]) + list(r["p10"][::-1]), fill="toself",
                             fillcolor="rgba(37,99,235,0.12)", line=dict(width=0),
                             name="simulation 10th to 90th percentile band (not a calibrated interval)"))
    fig.add_trace(go.Scatter(x=t, y=r["path"], line=dict(color=ACCENT, width=2), name="simulated path, no noise"))
    fig.add_hline(y=70, line=dict(color=WARN, dash="dot", width=1))
    fig.update_layout(title="Next 3 simulated days on current doses, with this person's stated assumptions"
                            + (" and planned fast dates" if change.fasting_dates else ""),
                      height=320, yaxis_title="Glucose (mg/dL)", margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor="white", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")
    hc = json.loads((ROOT / "data" / "synthetic" / "health_connect" / f"{pidp}.json").read_text())
    steps = pd.DataFrame(hc["StepsRecord"])
    steps["day"] = pd.to_datetime(steps["startTime"]).dt.date
    w1, w2 = st.columns(2)
    w1.metric("Steps a day (14-day median)", f"{int(steps.groupby('day')['count'].sum().median()):,}")
    w2.metric("Sleep sessions logged", len(hc["SleepSessionRecord"]))
    st.caption("Display-only synthetic wearable summary, in Android Health Connect record shapes. Not used by any model.")

# ---------------------------------------------------------------- dose scenarios

with tab_scen:
    st.subheader("Illustrative dose scenarios for clinician discussion")
    st.caption("Research feature. Each option is simulated on the persona's physiology core with stated assumptions. "
               "Causal treatment response and patient-specific risk have not been validated. No option is recommended; "
               "the doctor decides.")
    pid_s = st.selectbox("Patient ", [p["id"] for p in people], index=0, key="scen_pid", format_func=label)
    person = by_id[pid_s]
    default = body_change_for(person, as_of)
    a1, a2, a3, a4, a5 = st.columns(5)
    appetite = a1.slider("Meals eaten less on a GLP-1 (%)", 0, 50, int(round(default.appetite_cut * 100)), 5, key=f"ap_{pid_s}")
    sens_gain = a2.slider("Extra drug effect from weight loss (%)", 0, 40, int(round(default.sensitivity_gain * 100)), 5, key=f"sg_{pid_s}")
    half = a3.slider("Longer sulfonylurea action, kidneys (x)", 1.0, 2.5, float(default.su_half_life), 0.1, key=f"hl_{pid_s}")
    noise = a4.select_slider("Process-noise multiplier", options=list(NOISE_FACTORS), value=0.5, key=f"nf_{pid_s}")
    plan = person.get("fasting_plan")
    fast = a5.checkbox("Apply the planned fast on its dates", value=bool(default.fasting_dates), key=f"fa_{pid_s}",
                       disabled=not plan)
    change = BodyChange(appetite_cut=appetite / 100, sensitivity_gain=sens_gain / 100, su_half_life=half,
                        insulin_duration=default.insulin_duration, su_exposure=default.su_exposure,
                        fasting_hours=tuple(plan["window"]) if (plan and fast) else None,
                        fasting_dates=tuple(plan["dates"]) if (plan and fast) else ())
    tw = twin_for(person)
    opts = dose_options(person)
    start = as_of + pd.Timedelta(days=1)
    with st.spinner("Simulating each option..."):
        table = compare(tw, opts, change, n_sims=300, noise_factor=noise, start=start)
        before = compare(tw, {"Keep current doses": opts["Keep current doses"]},
                         BodyChange(su_exposure=default.su_exposure), n_sims=300, noise_factor=noise, start=start)
    st.markdown(f"Before the body changes, current doses gave **{before.iloc[0]['night_low_risk_percent']:.0f}%** simulated "
                f"nights with a low and **{before.iloc[0]['time_in_range_percent']:.0f}%** simulated time in range "
                f"(noise multiplier {noise}).")
    show = table.drop(columns=["noise_factor"]).rename(columns={
        "option": "Option", "night_low_risk_percent": "Simulated nights with a low (%)",
        "nights_simulated": "Nights simulated", "nights_with_a_low_in_14": "Expected nights with a low",
        "mean_glucose_mg_dl": "Mean glucose (mg/dL)", "time_in_range_percent": "Time 70 to 180 (%)",
        "time_above_180_percent": "Time above 180 (%)", "time_below_70_percent": "Time below 70 (%)"})
    st.dataframe(show, hide_index=True, width="stretch")


    @st.cache_data(show_spinner=False)
    def noise_grid(pid: str, key: str):
        person = by_id[pid]
        ch = body_change_for(person, as_of)
        return {nf: compare(twin_for(person), dose_options(person), ch, n_sims=300, noise_factor=nf,
                            start=start).set_index("option")["night_low_risk_percent"] for nf in NOISE_FACTORS}


    grid = noise_grid(pid_s, str(as_of.date()))
    st.markdown("**How much the answer depends on the noise assumption** (default assumptions; simulated nights with a low, %)")
    st.dataframe(pd.DataFrame({f"noise x{nf}": v for nf, v in grid.items()}), width="stretch")
    st.caption("More simulations reduce Monte Carlo noise; they do not make these numbers clinically calibrated. The "
               "noise level, the body-change effects and the model structure are all unvalidated assumptions.")
    for f in check(person, as_of):
        if f.kind in ("trigger", "safety"):
            st.caption(f"{f.name}: {f.note}")

# ---------------------------------------------------------------- evidence

with tab_evidence:
    st.subheader("Retrospective evaluation on 100 real adults with Type 2 diabetes (ShanghaiT2DM, open data)")
    d = ev["data"]
    st.caption(f"Protocol v2 (docs/evaluation_protocol.md): 5-day calibration per person, population models tested on "
               f"held-out patients (5 patient folds), no data from after a forecast's origin. {d['periods_scored']} "
               f"periods, {d['patients_scored']} patients, {d['monitored_patient_days']} monitored days. Same patients were "
               "inspected in an earlier, superseded evaluation, so this is development evidence, not independent validation.")
    rows = []
    for h in ("30 min", "60 min", "120 min"):
        for m in ("persistence", "trend", "physiology", "A", "B", "D", "C"):
            x = ev["forecast"][h][m]
            rows.append({"Horizon": h, "Method": x["name"], "RMSE (mg/dL)": f"{x['rmse_mg_dl']:.1f}",
                         "95% CI": f"{x['rmse_95ci'][0]:.1f} to {x['rmse_95ci'][1]:.1f}", "MARD (%)": x["mard_percent"],
                         "Clarke A+B (%)": x["clarke_a_plus_b_percent"]})
    st.markdown("**Glucose forecasts** (reference: CGM corrected by each person's frozen calibration offset)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    prow = []
    for h, comps in ev["paired_differences"].items():
        for lab_, x in comps.items():
            prow.append({"Horizon": h, "Comparison (RMSE difference, negative = first is better)": lab_,
                         "Difference (mg/dL)": f"{x['rmse_difference_mg_dl']:+.2f}",
                         "95% CI": f"{x['95ci'][0]:+.2f} to {x['95ci'][1]:+.2f}" if x["95ci"] else "n/a"})
    st.markdown("**What each ingredient adds** (paired, patient-level bootstrap)")
    st.dataframe(pd.DataFrame(prow), hide_index=True, width="stretch")
    arow = []
    for setting in ("budget_0.5", "budget_1.0", "fixed_85", "fixed_90"):
        for m, x in ev["alarms"].get(setting, {}).items():
            arow.append({"Setting": setting.replace("_", " "), "Alert score": x["name"],
                         "Lows caught": f"{x['caught']} of {x['eligible_events']}",
                         "Sensitivity (%)": x["sensitivity_percent"],
                         "95% CI": f"{x['sensitivity_95ci'][0]} to {x['sensitivity_95ci'][1]}" if x["sensitivity_95ci"] else "n/a",
                         "Alerts/day": f"{x['emitted_alerts_per_day']:.2f}", "False alerts/day": f"{x['false_alerts_per_day']:.2f}",
                         "Alert precision (%)": x["alert_precision_percent"],
                         "Median lead (min)": x["lead_min_median"]})
    st.markdown("**Low alerts** (CGM-defined episodes, at least 15 minutes below 70 mg/dL; thresholds for the budget "
                "settings were chosen on training patients only)")
    st.dataframe(pd.DataFrame(arow), hide_index=True, width="stretch")
    c = ev["classifier"]
    st.caption(f"Low-within-60-minutes classifier: AUROC {c['auroc']}, AUPRC {c['auprc']} at a prevalence of "
               f"{c['prevalence']:.2%}. A high AUROC alone is not useful for rare events.")
    so = ev.get("sensor_off")
    if so:
        st.markdown("**Experimental sensor-off glucose estimation; acute low alerts unavailable**")
        st.write(f"From the end of calibration, with only fingersticks, doses and meals: RMSE {so['hybrid']['rmse_mg_dl']:.1f} "
                 f"mg/dL (learning layer) vs {so['carried']['rmse_mg_dl']:.1f} (last fingerstick carried) and "
                 f"{so['typical']['rmse_mg_dl']:.1f} (the person's typical day), over {so['hours']:,.0f} hours in "
                 f"{so['patients']} patients. Only {so['hybrid']['share_of_readings_below_70_estimated_below_80_percent']}% "
                 f"of readings below 70 were estimated below 80. Coverage: median {so['days_scored_per_period']['median']} "
                 f"and at most {so['days_scored_per_period']['max']:.1f} days per period; longer sensor-off use is untested.")
    st.caption("Superseded version-1 results are kept in results/legacy_2026-10-19/ for the record and are not quoted.")

# ---------------------------------------------------------------- data formats

with tab_data:
    st.caption("Example record shapes only: FHIR R4 bundles modelled on the ABDM implementation guide and Android "
               "Health Connect record types. They have not been validated against those profiles; no compliance or "
               "certification is claimed.")
    pid_d = st.selectbox("Patient  ", [p["id"] for p in people], index=0, key="data_pid")
    st.markdown("**Clinic record (FHIR R4 bundle, example shape)**")
    st.json(json.loads((ROOT / "data" / "synthetic" / "fhir" / f"{pid_d}.json").read_text()), expanded=True)
    hc = json.loads((ROOT / "data" / "synthetic" / "health_connect" / f"{pid_d}.json").read_text())
    st.markdown("**Wearable and CGM (Health Connect record shapes, first records of each type; display only)**")
    st.json({k: (v[:2] if isinstance(v, list) else v) for k, v in hc.items()}, expanded=2)
