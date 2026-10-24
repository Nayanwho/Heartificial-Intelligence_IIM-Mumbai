"""Load ShanghaiT2DM into tidy, per-period records.

Each Excel file is one sensor period of one patient: CGM every 15 minutes,
fingersticks, meals, insulin and tablets, all time-stamped. The summary sheet adds
labs and history. Source: ShanghaiT2DM (Scientific Data, 2023), CC BY 4.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .drugs import classes_in_text, parse_insulin, parse_oral

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "shanghai_t2dm"


def hba1c_percent(mmol_per_mol: float) -> float:
    """Convert IFCC HbA1c (mmol/mol) to NGSP percent."""
    return mmol_per_mol / 10.929 + 2.15


@dataclass
class Period:
    """One sensor period of one patient."""

    period_id: str
    patient_id: str
    cgm: pd.Series  # mg/dL, indexed by time
    fingerstick: pd.Series  # mg/dL, indexed by time
    meals: pd.DataFrame  # time, text, items [(food, amount, unit)], logged
    insulin: pd.DataFrame  # time, product, units, route, components
    pump_bolus: pd.Series  # IU of regular insulin, indexed by time
    pump_basal: pd.Series  # IU/h of regular insulin, the rate set at each time
    tablets: pd.DataFrame  # time, drug, drug_class, dose_mg
    info: dict  # summary-sheet row

    @property
    def start(self) -> pd.Timestamp:
        return self.cgm.index.min()

    @property
    def days(self) -> float:
        return (self.cgm.index.max() - self.cgm.index.min()).total_seconds() / 86400.0

    def drug_classes(self, before: pd.Timestamp | None = None) -> set[str]:
        """Drug classes in the record. With `before`, only dose records timestamped earlier count.

        The summary sheet's agent list carries no dates; it is treated as the baseline record
        (see docs/evaluation_protocol.md, section 4)."""
        tablets, insulin = self.tablets, self.insulin
        bolus, basal = self.pump_bolus, self.pump_basal
        if before is not None:
            cut = pd.Timestamp(before)
            tablets = tablets[tablets["time"] < cut] if len(tablets) else tablets
            insulin = insulin[insulin["time"] < cut] if len(insulin) else insulin
            bolus = bolus[bolus.index < cut] if len(bolus) else bolus
            basal = basal[basal.index < cut] if len(basal) else basal
        found = set(tablets["drug_class"]) if len(tablets) else set()
        if len(insulin) or len(bolus) or len(basal):
            found.add("insulin")
        return found | classes_in_text(self.info.get("agents", ""))

    def until(self, t: pd.Timestamp) -> "Period":
        """The period as it was known at time `t`: every record timestamped after `t` removed.

        Used by the live replay and by the future-data tests, so later data physically cannot
        reach a computation made at `t`."""
        cut = pd.Timestamp(t)

        def rows(df: pd.DataFrame) -> pd.DataFrame:
            return df[df["time"] <= cut].reset_index(drop=True) if len(df) else df

        def series(s: pd.Series) -> pd.Series:
            return s[s.index <= cut] if len(s) else s

        return Period(period_id=self.period_id, patient_id=self.patient_id, cgm=series(self.cgm),
                      fingerstick=series(self.fingerstick), meals=rows(self.meals), insulin=rows(self.insulin),
                      pump_bolus=series(self.pump_bolus), pump_basal=series(self.pump_basal),
                      tablets=rows(self.tablets), info=self.info)


def _find(df: pd.DataFrame, *keys: str) -> str | None:
    for col in df.columns:
        name = str(col).strip().lower()
        if any(k in name for k in keys):
            return col
    return None


def _numeric(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[col], errors="coerce")
    s.index = df["time"]
    return s.dropna().sort_index()


def parse_meal(text: object) -> tuple[list[tuple[str, float | None, str | None]], bool]:
    """Return ([(food, amount, unit)], logged). 'data not available' means a meal with unknown content."""
    t = str(text).replace("\xa0", " ").strip()
    if re.fullmatch(r"(?i)data not available", t):
        return [], False
    items = []
    for line in re.split(r"[\n;]+", t):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(.*?)\s*(\d+(?:\.\d+)?)\s*(g|ml|kg)\b", line, flags=re.I)
        if m and m.group(1).strip():
            amount = float(m.group(2)) * (1000.0 if m.group(3).lower() == "kg" else 1.0)
            unit = "ml" if m.group(3).lower() == "ml" else "g"
            items.append((m.group(1).strip().lower(), amount, unit))
        else:
            items.append((line.lower(), None, None))
    return items, True


def load_summary(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    s = pd.read_excel(data_dir / "Shanghai_T2DM_Summary.xlsx")
    s.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in s.columns]
    rename = {
        "Patient Number": "period_id",
        "Gender (Female=1, Male=2)": "sex",
        "Age (years)": "age",
        "Height (m)": "height_m",
        "Weight (kg)": "weight_kg",
        "BMI (kg/m2)": "bmi",
        "Duration of Diabetes (years)": "duration_years",
        "Hypoglycemic Agents": "agents",
        "Other Agents": "other_agents",
        "HbA1c (mmol/mol)": "hba1c_mmol_mol",
        "Estimated Glomerular Filtration Rate (ml/min/1.73m2)": "egfr",
        "Creatinine (umol/L)": "creatinine_umol_l",
        "Fasting C-peptide (nmol/L)": "c_peptide_fasting",
        "Hypoglycemia (yes/no)": "hypo_history",
        "Diabetic Microvascular Complications": "microvascular",
        "Comorbidities": "comorbidities",
    }
    s = s.rename(columns=rename)
    s["period_id"] = s["period_id"].astype(str).str.strip()
    s["patient_id"] = s["period_id"].str.slice(0, 4)
    s["sex"] = s["sex"].map({1: "F", 2: "M"})
    for col in ("age", "bmi", "weight_kg", "egfr", "hba1c_mmol_mol", "duration_years", "c_peptide_fasting"):
        s[col] = pd.to_numeric(s[col], errors="coerce")
    s["hba1c_percent"] = s["hba1c_mmol_mol"].map(lambda v: round(hba1c_percent(v), 1) if pd.notna(v) else np.nan)
    s["hypo_history"] = s["hypo_history"].astype(str).str.strip().str.lower().eq("yes")
    return s


def load_period(path: Path, info: dict | None = None) -> Period:
    df = pd.read_excel(path)
    date_col = _find(df, "date")
    df["time"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    cgm = _numeric(df, _find(df, "cgm"))
    fingerstick = _numeric(df, _find(df, "cbg"))
    pump_bolus = _numeric(df, _find(df, "bolus"))
    pump_basal = _numeric(df, _find(df, "basal", "iu / h", "基础"))

    meal_rows = []
    meal_col = _find(df, "dietary")
    if meal_col is not None:
        for time, text in df[["time", meal_col]].dropna().itertuples(index=False):
            items, logged = parse_meal(text)
            meal_rows.append({"time": time, "text": str(text), "items": items, "logged": logged})
    meals = pd.DataFrame(meal_rows, columns=["time", "text", "items", "logged"])

    insulin_rows = []
    for key, route in (("insulin dose - s.c", "sc"), ("i.v", "iv")):
        col = _find(df, key)
        if col is None:
            continue
        for time, text in df[["time", col]].dropna().itertuples(index=False):
            for d in parse_insulin(text, route=route):
                insulin_rows.append({"time": time, "product": d.product, "units": d.units,
                                     "route": route, "components": d.components})
    insulin = pd.DataFrame(insulin_rows, columns=["time", "product", "units", "route", "components"])

    tablet_rows = []
    oral_col = _find(df, "non-insulin")
    if oral_col is not None:
        for time, text in df[["time", oral_col]].dropna().itertuples(index=False):
            for d in parse_oral(text):
                tablet_rows.append({"time": time, "drug": d.drug, "drug_class": d.drug_class, "dose_mg": d.dose_mg})
    tablets = pd.DataFrame(tablet_rows, columns=["time", "drug", "drug_class", "dose_mg"])

    period_id = path.stem
    return Period(period_id=period_id, patient_id=period_id[:4], cgm=cgm, fingerstick=fingerstick,
                  meals=meals, insulin=insulin, pump_bolus=pump_bolus, pump_basal=pump_basal,
                  tablets=tablets, info=info or {})


@lru_cache(maxsize=2)
def load_all(data_dir: str = str(DATA_DIR)) -> tuple[Period, ...]:
    """Every sensor period, with its summary row attached as `info`."""
    root = Path(data_dir)
    summary = load_summary(root).set_index("period_id")
    periods = []
    for path in sorted((root / "periods").glob("*.xls*")):
        info = summary.loc[path.stem].to_dict() if path.stem in summary.index else {}
        periods.append(load_period(path, info))
    return tuple(periods)
