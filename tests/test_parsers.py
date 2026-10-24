import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from twin.data_shanghai import hba1c_percent, parse_meal  # noqa: E402
from twin.drugs import classes_in_text, parse_insulin, parse_oral  # noqa: E402


def test_premix_and_basal_in_one_record():
    doses = parse_insulin("insulin glargine, 14 IU, Humulin 70/30, 12 IU")
    assert [(d.units, d.components) for d in doses] == [
        (14.0, {"long": 1.0}),
        (12.0, {"regular": 0.3, "nph": 0.7}),
    ]


def test_analogue_premix_is_not_read_as_rapid():
    (dose,) = parse_insulin("insulin aspart 70/30, 14 IU")
    assert dose.components == {"rapid": 0.3, "nph": 0.7}


def test_typos_and_missing_commas():
    assert parse_insulin("insulin glarigine, 12 IU")[0].components == {"long": 1.0}
    assert parse_insulin("Humulin 70/30  8 IU")[0].units == 8.0
    assert parse_insulin("insulin\xa0glargine, 18 IU")[0].components == {"long": 1.0}


def test_intravenous_record_names_product_after_dose():
    (dose,) = parse_insulin("500ml 0.9% sodium chloride, 15 IU Novolin R, 10 ml 10% potassium chloride",
                            route="iv")
    assert dose.units == 15.0 and dose.components == {"iv": 1.0}


def test_oral_units_and_slips():
    doses = {d.drug: d.dose_mg for d in parse_oral("metformin 0.5 g, glimepiride 2 mg")}
    assert doses == {"metformin": 500.0, "glimepiride": 2.0}
    assert parse_oral("\xa0voglibose 0.2 g")[0].dose_mg == 0.2  # 'g' written for 'mg'
    assert parse_oral("linagliptin 0.1 g")[0].dose_mg is None  # implausible either way


def test_fixed_dose_combination():
    doses = parse_oral("sitagliptinphosphate/metforminhydrochloride tablets 50 mg / 500 mg")
    assert [(d.drug, d.dose_mg) for d in doses] == [("sitagliptin", 50.0), ("metformin", 500.0)]


def test_drug_classes_from_summary_text():
    assert classes_in_text("metformin, Humulin 70/30") == {"metformin", "insulin"}
    assert classes_in_text("acarbose, dapagliflozin, gliclazide") == {"agi", "sglt2", "sulfonylurea"}
    assert classes_in_text("liraglutide 1.2 mg") == {"glp1"}


def test_meal_lines_and_missing_meals():
    items, logged = parse_meal("Rice 150 g\nVegetable 100 g\nSoybean milk 250 ml")
    assert logged and items == [("rice", 150.0, "g"), ("vegetable", 100.0, "g"), ("soybean milk", 250.0, "ml")]
    assert parse_meal("data not available") == ([], False)
    assert parse_meal("Seafood porridge 1200g")[0] == [("seafood porridge", 1200.0, "g")]


def test_hba1c_conversion():
    assert round(hba1c_percent(75.9), 1) == 9.1
