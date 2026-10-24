"""Regenerate every result, summary number and figure, in order (about 15 to 20 minutes).

    python scripts/run_all.py

Runs, from the repository root: the descriptive sensor check, the version-2 evaluation (both
references), the synthetic cohort, the scenario sensitivity grid, the summary used by the documents
(it also rewrites the README's result tables) and the figures. Then run `python -m pytest -q tests`.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["data_check.py", "evaluate.py", "make_cohort.py", "scenario_sensitivity.py", "summarise.py", "make_figures.py"]

if __name__ == "__main__":
    t0 = time.time()
    for step in STEPS:
        print(f"== {step}", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / step)], cwd=ROOT, check=True)
    print(f"all steps done in {time.time() - t0:.0f} s; now run: python -m pytest -q tests")
