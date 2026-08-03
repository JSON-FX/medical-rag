from evals.axes import NEAR_MISS_AXES, verified_absent_axes


def test_axis_present_when_any_keyword_appears():
    text = "Pediatric Dosage: starting dose 500 mg orally twice a day."
    assert "pediatric" not in verified_absent_axes(text)


def test_axis_absent_when_no_keyword_appears():
    text = "Adult dosage: 50 mg once daily for hypertension."
    absent = verified_absent_axes(text)
    assert "pediatric" in absent
    assert "pregnancy" in absent


def test_scan_is_case_insensitive():
    assert "pregnancy" not in verified_absent_axes("PREGNANT women should not take this.")


def test_every_axis_is_reported_one_way_or_the_other():
    absent = set(verified_absent_axes("some unrelated text"))
    assert absent == set(NEAR_MISS_AXES), "an axis went unreported"


def test_real_label_phrasings_are_detected():
    """These exact phrasings appear in the pinned FDA labels."""
    cases = [
        ("pediatric", "In Pediatric Patients over 3 Months of Age, 20 to 45 mg/kg/day"),
        ("pediatric", "The safety and effectiveness in children have not been established"),
        ("overdose", "Overdose of metformin hydrochloride has occurred"),
        ("pregnancy", "Limited data with metformin in pregnant women"),
        ("geriatric", "Atenolol is excreted by the kidneys; elderly patients"),
        ("hepatic", "No dosage adjustment is needed for hepatic impairment"),
    ]
    for axis, text in cases:
        assert axis not in verified_absent_axes(text), f"{axis!r} not detected in {text!r}"


def test_absent_axes_are_sorted_for_stable_manifests():
    assert verified_absent_axes("x") == sorted(verified_absent_axes("x"))
