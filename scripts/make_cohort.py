"""Generate the synthetic Indian demo cohort (FHIR bundles and Health Connect files).

Run from the repository root after scripts/evaluate.py:  python scripts/make_cohort.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from twin.synthetic import write_cohort  # noqa: E402

if __name__ == "__main__":
    twins = pd.read_csv(ROOT / "results" / "twins.csv")
    summary = write_cohort(twins, ROOT / "data" / "synthetic")
    print(summary[["id", "persona", "age", "egfr_now", "weight_change_percent", "cyp2c9", "g6pd", "fasting_plan"]].to_string(index=False))
