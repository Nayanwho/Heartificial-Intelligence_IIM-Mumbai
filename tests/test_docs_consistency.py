"""Every headline number in the public documents must match results/summary.json.

The documents quote values exactly as summary.json formats them. After a rerun that changes a
number, this test fails until the README, deck, architecture and video script are updated.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary.json"
SCRIPT = ROOT.parent / "santulan_handover" / "VIDEO_SCRIPT.md"  # kept outside the public repository

DOCS = {
    "README.md": ["patients_scored", "periods_scored", "hybrid_rmse_30", "hybrid_rmse_60", "hybrid_rmse_120",
                  "hybrid_rmse_120_ci", "persistence_rmse_120", "gain_120_percent", "physiology_value_120",
                  "physiology_value_120_ci", "physiology_value_120_with_record", "physiology_value_120_with_record_ci",
                  "record_value_30", "record_value_60", "events", "event_patients", "current_caught_05", "hybrid_caught_05",
                  "current_caught_10", "hybrid_caught_10", "current_false_per_day_05", "raw_events",
                  "raw_current_caught_05", "raw_hybrid_caught_05", "classifier_auroc", "classifier_auprc",
                  "sensor_off_rmse", "sensor_off_typical_rmse", "sensor_off_carried_rmse", "sensor_off_low_readings",
                  "sensor_off_max_days", "raw_lows", "corrected_lows", "confirmed_lows", "meena_keep_05",
                  "meena_halve_evening_05", "meena_halve_evening_noise_range", "meena_halve_evening_kidney_max",
                  "meena_keep_no_kidney"],
    "docs/presentation.html": ["patients_scored", "hybrid_rmse_120", "persistence_rmse_120", "gain_120_percent",
                               "physiology_value_120", "physiology_value_120_with_record", "current_caught_05", "hybrid_caught_05", "events",
                               "sensor_off_rmse", "sensor_off_low_readings", "sensor_off_max_days", "corrected_lows",
                               "meena_halve_evening_noise_range", "meena_halve_evening_kidney_max"],
    "docs/architecture.html": ["hybrid_rmse_30", "hybrid_rmse_60", "hybrid_rmse_120", "current_caught_05",
                               "sensor_off_rmse", "sensor_off_max_days"],
}


@pytest.mark.skipif(not SUMMARY.exists(), reason="run scripts/summarise.py first")
@pytest.mark.parametrize("doc", sorted(DOCS))
def test_document_quotes_current_results(doc):
    summary = json.loads(SUMMARY.read_text())
    text = (ROOT / doc).read_text(encoding="utf-8")
    missing = [f"{k}={summary[k]!r}" for k in DOCS[doc] if summary[k] not in text]
    assert not missing, f"{doc} does not quote: {missing}"


@pytest.mark.skipif(not SUMMARY.exists() or not SCRIPT.exists(), reason="video script not present")
def test_video_script_quotes_current_results():
    summary = json.loads(SUMMARY.read_text())
    text = SCRIPT.read_text(encoding="utf-8")
    keys = ["hybrid_rmse_120", "persistence_rmse_120", "current_caught_05", "hybrid_caught_05", "sensor_off_rmse",
            "physiology_value_120", "physiology_value_120_with_record", "record_value_30", "record_value_60"]
    missing = [f"{k}={summary[k]!r}" for k in keys if summary[k] not in text]
    assert not missing, f"VIDEO_SCRIPT.md does not quote: {missing}"
    s = summary
    assert f"errors are {s['hybrid_rmse_30']} and {s['hybrid_rmse_60']}" in text


@pytest.mark.skipif(not SUMMARY.exists(), reason="run scripts/summarise.py first")
def test_composite_quotes_match():
    """Pairs of numbers quoted together, where a single-string check could pass by accident."""
    s = json.loads(SUMMARY.read_text())
    deck = (ROOT / "docs" / "presentation.html").read_text(encoding="utf-8")
    arch = (ROOT / "docs" / "architecture.html").read_text(encoding="utf-8")
    assert f"{s['hybrid_rmse_30']} / {s['hybrid_rmse_60']}</div>" in deck
    assert f"RMSE {s['hybrid_rmse_30']} / {s['hybrid_rmse_60']} / {s['hybrid_rmse_120']} mg/dL" in arch


@pytest.mark.skipif(not (ROOT / "results" / "evaluation.json").exists(), reason="run scripts/evaluate.py first")
def test_readme_result_tables_are_the_generated_ones():
    sys.path.insert(0, str(ROOT / "scripts"))
    import summarise
    ev = json.loads((ROOT / "results" / "evaluation.json").read_text())
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert summarise.forecast_table(ev) in readme
    assert summarise.alert_table(ev) in readme


def test_no_superseded_headline_remains():
    stale = ["50 of 54", "93%", "35 lives", "2,900", "two in three raw", "33.2 mg/dL", "0.53 false"]
    for doc in DOCS:
        text = (ROOT / doc).read_text(encoding="utf-8")
        # the README may name the superseded numbers once, when explaining what changed
        allowed = text.count("superseded") > 0 and doc == "README.md"
        hits = [s for s in stale if s in text and not (allowed and s in ("50 of 54", "33.2"))]
        assert not hits, f"{doc} still contains superseded claims: {hits}"
