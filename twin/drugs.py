"""Drug knowledge used by the parsers and the physiology core.

Insulin products are described as mixtures of five action profiles. Every other
glucose-lowering drug is mapped to its class. Sulfonylureas carry the facts the
twin needs for their exposure model. All values are typical label or textbook
figures; the twin refines the exposure scale per patient during calibration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Insulin
# ---------------------------------------------------------------------------

# Typical time to peak action and total duration, in hours.
INSULIN_PROFILES: dict[str, dict[str, float]] = {
    "rapid": {"peak_h": 1.0, "duration_h": 4.5},  # aspart, lispro, glulisine
    "regular": {"peak_h": 2.5, "duration_h": 7.0},  # human regular insulin (R)
    "nph": {"peak_h": 6.0, "duration_h": 14.0},  # human NPH and protaminated analogues
    "long": {"peak_h": 10.0, "duration_h": 24.0},  # glargine, detemir: broad, near flat
    "ultralong": {"peak_h": 12.0, "duration_h": 42.0},  # degludec: flatter still
    "iv": {"peak_h": 0.1, "duration_h": 1.0},  # intravenous regular insulin
}

# Product patterns, most specific first: premixes before their components.
INSULIN_PRODUCTS: list[tuple[str, dict[str, float]]] = [
    (r"aspart\s*70\s*/\s*30|novomix\s*30", {"rapid": 0.3, "nph": 0.7}),
    (r"aspart\s*50\s*/\s*50|novomix\s*50", {"rapid": 0.5, "nph": 0.5}),
    (r"lispro\s*(?:mix\s*)?25|humalog\s*mix\s*25", {"rapid": 0.25, "nph": 0.75}),
    (r"lispro\s*(?:mix\s*)?50|humalog\s*mix\s*50", {"rapid": 0.5, "nph": 0.5}),
    (r"aspart|novorapid|lispro|humalog|glulisine|apidra", {"rapid": 1.0}),
    (r"novolin\s*30\s*r|humulin\s*70\s*/\s*30|scilin\s*m\s*30", {"regular": 0.3, "nph": 0.7}),
    (r"gansulin\s*40\s*r", {"regular": 0.4, "nph": 0.6}),
    (r"novolin\s*50\s*r", {"regular": 0.5, "nph": 0.5}),
    (r"novolin\s*r|gansulin\s*r|humulin\s*r", {"regular": 1.0}),
    (r"novolin\s*n|humulin\s*n|\bnph\b|isophane", {"nph": 1.0}),
    (r"glar[a-z]*ine|lantus|basaglar|toujeo|detemir|levemir", {"long": 1.0}),
    (r"degludec|tresiba", {"ultralong": 1.0}),
]


@dataclass(frozen=True)
class InsulinDose:
    product: str
    units: float
    components: dict[str, float] = field(hash=False)


def _clean(text: object) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\xa0", " ")).strip().lower()


def _match_insulin(segment: str) -> tuple[dict[str, float] | None, str]:
    for pattern, components in INSULIN_PRODUCTS:
        m = re.search(pattern, segment)
        if m:
            return components, m.group(0)
    return None, ""


def parse_insulin(text: object, route: str = "sc") -> list[InsulinDose]:
    """Parse an injection record such as 'insulin glargine, 14 IU, Humulin 70/30, 12 IU'.

    The product name usually comes before its dose; intravenous records put it after
    ('15 IU Novolin R'), so the text after the dose is tried when the text before has none.
    """
    t = _clean(text)
    doses: list[InsulinDose] = []
    matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*iu\b", t))
    prev_end = 0
    for i, m in enumerate(matches):
        before = t[prev_end : m.start()]
        after_end = matches[i + 1].start() if i + 1 < len(matches) else len(t)
        after = t[m.end() : after_end].split(",")[0]
        prev_end = m.end()
        components, name = _match_insulin(before)
        if components is None:
            components, name = _match_insulin(after)
        if components is None:
            continue
        if route == "iv":
            components, name = {"iv": 1.0}, f"{name} (iv)"
        doses.append(InsulinDose(name, float(m.group(1)), components))
    return doses


# ---------------------------------------------------------------------------
# Other glucose-lowering drugs
# ---------------------------------------------------------------------------

DRUG_CLASSES: dict[str, list[str]] = {
    "sulfonylurea": ["gliclazide", "glimepiride", "glipizide", "gliquidone", "glibenclamide", "glyburide"],
    "glinide": ["repaglinide", "nateglinide", "mitiglinide"],
    "metformin": ["metformin"],
    "agi": ["acarbose", "voglibose", "miglitol"],
    "dpp4": ["sitagliptin", "linagliptin", "saxagliptin", "vildagliptin", "alogliptin"],
    "sglt2": ["dapagliflozin", "empagliflozin", "canagliflozin", "ertugliflozin", "henagliflozin"],
    "tzd": ["pioglitazone", "rosiglitazone"],
    "glp1": ["liraglutide", "semaglutide", "dulaglutide", "exenatide", "lixisenatide",
             "loxenatide", "beinaglutide", "tirzepatide"],
}
DRUG_TO_CLASS = {drug: cls for cls, drugs in DRUG_CLASSES.items() for drug in drugs}

# Plausible single doses in mg; a record outside the range is kept but its dose is set to None.
PLAUSIBLE_MG: dict[str, tuple[float, float]] = {
    "gliclazide": (15, 160), "glimepiride": (0.5, 8), "glipizide": (2.5, 20),
    "gliquidone": (15, 60), "glibenclamide": (1.25, 10), "glyburide": (1.25, 10),
    "repaglinide": (0.5, 4), "nateglinide": (60, 180), "mitiglinide": (5, 20),
    "metformin": (250, 1000), "acarbose": (25, 100), "voglibose": (0.1, 0.3), "miglitol": (25, 100),
    "sitagliptin": (25, 100), "linagliptin": (5, 5), "saxagliptin": (2.5, 5),
    "vildagliptin": (50, 50), "alogliptin": (6.25, 25),
    "dapagliflozin": (5, 10), "empagliflozin": (10, 25), "canagliflozin": (100, 300),
    "ertugliflozin": (5, 15), "henagliflozin": (5, 10),
    "pioglitazone": (15, 45), "rosiglitazone": (2, 8),
    "liraglutide": (0.6, 3), "semaglutide": (0.25, 14), "dulaglutide": (0.75, 4.5),
    "exenatide": (0.005, 2), "lixisenatide": (0.01, 0.02), "loxenatide": (0.1, 0.2),
    "beinaglutide": (0.1, 0.2), "tirzepatide": (2.5, 15),
}

# Sulfonylurea facts used by the exposure model (typical label values).
SULFONYLUREAS: dict[str, dict[str, object]] = {
    "glimepiride": {"half_life_h": 7.0, "main_enzyme": "CYP2C9"},  # label: 5 to 9 h
    "gliclazide": {"half_life_h": 12.0, "main_enzyme": "CYP2C9 and CYP2C19"},
    "glipizide": {"half_life_h": 4.0, "main_enzyme": "CYP2C9"},
    "gliquidone": {"half_life_h": 1.5, "main_enzyme": "hepatic, biliary excretion"},
    "glibenclamide": {"half_life_h": 10.0, "main_enzyme": "CYP2C9 and CYP3A4"},
    "glyburide": {"half_life_h": 10.0, "main_enzyme": "CYP2C9 and CYP3A4"},
}


@dataclass(frozen=True)
class DrugDose:
    drug: str
    drug_class: str
    dose_mg: float | None
    raw: str


def _to_mg(value: float, unit: str) -> float:
    if unit == "g":
        return value * 1000.0
    if unit in ("ug", "mcg", "µg"):
        return value / 1000.0
    return value


def _plausible(drug: str, mg: float) -> bool:
    lo, hi = PLAUSIBLE_MG.get(drug, (0.0, float("inf")))
    return lo * 0.5 <= mg <= hi * 1.5


def _dose_for(drug: str, value: float, unit: str) -> float | None:
    mg = _to_mg(value, unit)
    if _plausible(drug, mg):
        return mg
    # Common slip in the records: 'g' written for 'mg' (e.g. 'voglibose 0.2 g').
    if unit == "g" and _plausible(drug, value):
        return value
    return None


def parse_oral(text: object) -> list[DrugDose]:
    """Parse a non-insulin record such as 'metformin 500 mg, glimepiride 2 mg'.

    Handles fixed-dose combinations written as 'a/b tablets 50 mg / 500 mg'.
    """
    t = _clean(text)
    out: list[DrugDose] = []
    combo = re.search(r"([a-z]+)\s*/\s*([a-z]+).*?(\d+(?:\.\d+)?)\s*mg\s*/\s*(\d+(?:\.\d+)?)\s*mg", t)
    if combo:
        names = [n for n in (_known_drug(combo.group(1)), _known_drug(combo.group(2))) if n]
        values = [float(combo.group(3)), float(combo.group(4))]
        if len(names) == 2:
            for name, value in zip(names, values):
                out.append(DrugDose(name, DRUG_TO_CLASS[name], _dose_for(name, value, "mg"), t))
            return out
    for part in re.split(r"[,;，]", t):
        name = _known_drug(part)
        if not name:
            continue
        m = re.search(r"(\d+(?:\.\d+)?)\s*(mg|g|ug|mcg|µg)\b", part)
        dose = _dose_for(name, float(m.group(1)), m.group(2)) if m else None
        out.append(DrugDose(name, DRUG_TO_CLASS[name], dose, part.strip()))
    return out


def _known_drug(text: str) -> str | None:
    for drug in DRUG_TO_CLASS:
        if drug in text:
            return drug
    return None


def classes_in_text(text: object) -> set[str]:
    """Drug classes named anywhere in a free-text list such as a summary's agents column."""
    t = _clean(text)
    found = {cls for drug, cls in DRUG_TO_CLASS.items() if drug in t}
    components, _ = _match_insulin(t)
    if components is not None or "insulin" in t:
        found.add("insulin")
    return found
