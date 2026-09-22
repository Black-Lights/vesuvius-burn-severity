"""Checks on the constants in aoi.py: the rules every later step relies on."""

from burnsev import aoi


def test_bbox_is_west_south_east_north():
    west, south, east, north = aoi.BBOX
    assert west < east
    assert south < north


def test_windows_do_not_overlap_the_fire():
    # ISO date strings compare correctly as text.
    assert aoi.PRE_WINDOW[1] < aoi.FIRE_START
    assert aoi.POST_WINDOW[0] > aoi.FIRE_END


def test_severity_classes_are_contiguous_and_ordered():
    for (_, _, upper), (_, lower, _) in zip(aoi.SEVERITY_CLASSES, aoi.SEVERITY_CLASSES[1:]):
        assert upper == lower


def test_severe_classes_follow_the_cut():
    assert aoi.SEVERE_CLASSES == ("moderate-low", "moderate-high", "high")
    assert aoi.SEVERITY_CUT_DNBR == aoi.SEVERITY_CLASSES[2][1]


def test_sensitivity_includes_the_chosen_slope():
    assert aoi.SLOPE_THRESHOLD_DEG in aoi.SLOPE_SENSITIVITY_DEG
