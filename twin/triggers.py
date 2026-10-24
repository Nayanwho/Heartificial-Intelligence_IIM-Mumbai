"""Clinician-review triggers: moments when a dose that was right can become too strong.

Rules read only what a clinic record and a phone already hold, as of a given date (`as_of`,
injected; the demo uses 20 October 2026 and shows it). When one fires for someone on a
sulfonylurea or insulin, the dashboard flags the record for clinician review and offers the
illustrative scenario comparison. The rules are simple, documented policies, not validated
clinical decision rules; the label notes quote or paraphrase the drug labels; the doctor
decides.

Missing data stay missing: an untested genotype or G6PD status raises no safety flag and is
shown as "not tested".
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

DEMO_AS_OF = pd.Timestamp("2026-10-20")
GLP1 = ("semaglutide", "liraglutide", "dulaglutide", "tirzepatide", "exenatide")
SULFONYLUREAS = ("glimepiride", "gliclazide", "glipizide", "glibenclamide", "gliquidone")
LOWERING = SULFONYLUREAS + ("insulin",)
RECENT_START_WEEKS = 16
WEIGHT_LOSS_PERCENT = 5.0
FAST_LOOKAHEAD_DAYS = 14
EGFR_THRESHOLD = 60


@dataclass
class Flag:
    name: str
    kind: str  # "trigger", "safety" or "missing"
    reason: str
    note: str
    field: str = ""  # which record field caused it


def _date(x) -> str:
    return pd.Timestamp(x).strftime("%d %b %Y")


def check(person: dict, as_of: pd.Timestamp | None = None) -> list[Flag]:
    as_of = pd.Timestamp(as_of) if as_of is not None else DEMO_AS_OF
    meds = person["medicines"]
    names = [m[0] for m in meds]
    on_lowering = any(n.startswith(LOWERING) for n in names)
    on_su = any(n in SULFONYLUREAS for n in names)
    flags: list[Flag] = []

    for name, _dose, _how, start in meds:
        if name in GLP1 and on_lowering:
            started = pd.Timestamp(start)
            if started > as_of:
                continue  # a planned future start is not a recent start
            weeks = (as_of - started).days / 7
            if weeks <= RECENT_START_WEEKS:
                flags.append(Flag("GLP-1 start", "trigger",
                                  f"{name} started {_date(started)} ({weeks:.0f} weeks ago), alongside a sulfonylurea or insulin",
                                  "GLP-1 labels advise considering a lower sulfonylurea or insulin dose to reduce the risk of lows.",
                                  "medicines"))

    w_then, w_now = person.get("weight_kg_12w_ago"), person.get("weight_kg_now")
    if on_lowering and w_then and w_now:
        loss = 100 * (1 - w_now / w_then)
        if loss >= WEIGHT_LOSS_PERCENT:
            dates = person.get("weight_dates") or [None, None]
            when = f" ({_date(dates[0])} to {_date(dates[1])})" if dates[0] and dates[1] else ""
            flags.append(Flag("Weight loss", "trigger", f"{loss:.1f}% of body weight lost{when}: {w_then} to {w_now} kg",
                              "Weight loss raises insulin sensitivity, so the same dose may lower glucose further.",
                              "weight"))

    plan = person.get("fasting_plan") or {}
    upcoming = [pd.Timestamp(d) for d in plan.get("dates", [])
                if as_of.normalize() <= pd.Timestamp(d) <= as_of.normalize() + pd.Timedelta(days=FAST_LOOKAHEAD_DAYS)]
    if on_lowering and upcoming:
        first = min(upcoming)
        flags.append(Flag("Planned fast", "trigger",
                          f"{plan.get('label', 'fast')} on {_date(first)} (in {(first - as_of.normalize()).days} days)",
                          "On fasting days meals move or are skipped but doses often stay the same; the scenario "
                          "applies the skipped meals on those dates only.", "fasting_plan"))

    egfr = person.get("egfr_now")
    if egfr is None:
        flags.append(Flag("eGFR not recorded", "missing", "no kidney-function result on file",
                          "Kidney function is needed before judging sulfonylurea or insulin doses.", "egfr"))
    elif on_lowering and egfr < EGFR_THRESHOLD:
        dates = person.get("egfr_dates") or [None, None]
        now = f" on {_date(dates[1])}" if dates[1] else ""
        prev = person.get("egfr_prev")
        before = f"; previous {prev}{' on ' + _date(dates[0]) if dates[0] else ''}" if prev is not None else ""
        flags.append(Flag("Kidney function", "trigger", f"eGFR {egfr}{now}{before}",
                          "The glimepiride label advises a 1 mg starting dose and conservative titration in kidney "
                          "impairment; its metabolites accumulate as kidney function falls.", "egfr"))

    if on_su:
        g6pd = person.get("g6pd", "not tested")
        if g6pd == "deficient":
            flags.append(Flag("G6PD deficiency", "safety", "G6PD deficient and on a sulfonylurea",
                              "Sulfonylureas can cause haemolytic anaemia in G6PD deficiency; the glimepiride label "
                              "advises caution and considering a non-sulfonylurea alternative.", "g6pd"))
        elif g6pd == "not tested":
            flags.append(Flag("G6PD not tested", "missing", "G6PD status unknown", "Shown for completeness; no rule fires.",
                              "g6pd"))
        geno = person.get("cyp2c9", "not tested")
        if geno not in ("*1/*1", "not tested"):
            flags.append(Flag("CYP2C9 variant", "safety", f"CYP2C9 {geno}: slower sulfonylurea clearance",
                              "Literature links CYP2C9*3 to higher sulfonylurea exposure (evidence mixed); the "
                              "scenario raises exposure as a stated assumption.", "cyp2c9"))
        elif geno == "not tested":
            flags.append(Flag("CYP2C9 not tested", "missing", "genotype unknown",
                              "No genotype is assumed; the scenario uses standard exposure.", "cyp2c9"))
        if person.get("age", 0) >= 65:
            flags.append(Flag("Older adult", "safety", f"Age {person['age']} on a sulfonylurea",
                              "The glimepiride label advises conservative dosing in older patients, who are at higher "
                              "risk of lows.", "age"))
    return flags
