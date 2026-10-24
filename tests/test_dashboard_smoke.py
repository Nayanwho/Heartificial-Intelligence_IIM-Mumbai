"""Streamlit smoke run: the dashboard renders every tab and a scripted replay without errors."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "dashboard.py"

pytest.importorskip("streamlit")
needs_results = pytest.mark.skipif(not (ROOT / "results" / "models").exists() or not (ROOT / "results" / "twins.csv").exists(),
                                   reason="run scripts/evaluate.py and scripts/make_cohort.py first")


@needs_results
def test_dashboard_runs_a_scripted_replay():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.run()
    assert not at.exception
    jump = next(s for s in at.selectbox if s.label.startswith("Jump to"))
    night = next(o for o in jump.options if o.startswith("21 Nov 03:52"))
    jump.select(night).run()
    assert not at.exception
    assert any("Replay clock** 21 Nov 2020 02:37" in m.value for m in at.markdown)
    at.button(key=next(b.key for b in at.button if b.label == "+15 min")).click().run()
    assert any(m.value == "YES" for m in at.metric if m.label == "Alert now")
    next(c for c in at.checkbox if c.label.startswith("Reveal")).check().run()
    assert not at.exception
    assert any("scored matched" in c.value for c in at.caption)
