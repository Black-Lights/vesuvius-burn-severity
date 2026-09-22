"""Pure-function checks on catalog.py: the offset rule and the duplicate rule. No network."""

from datetime import UTC, datetime

import pystac

from burnsev import catalog

T0 = datetime(2025, 6, 7, 9, 50, 41, tzinfo=UTC)
T1 = datetime(2025, 6, 10, 9, 50, 29, tzinfo=UTC)


def _item(item_id: str, when: datetime, **props) -> pystac.Item:
    return pystac.Item(id=item_id, geometry=None, bbox=None, datetime=when, properties=dict(props))


def test_offset_follows_the_processing_baseline():
    assert catalog.boa_offset(_item("new", T0, **{"s2:processing_baseline": "05.11"})) == -1000
    assert catalog.boa_offset(_item("old", T0, **{"s2:processing_baseline": "03.01"})) == 0
    assert catalog.boa_offset(_item("unknown", T0)) == 0


def test_offset_prefers_raster_bands_metadata_when_published():
    item = _item("rb", T0, **{"s2:processing_baseline": "05.11"})
    item.add_asset(
        "B04", pystac.Asset(href="B04.tif", extra_fields={"raster:bands": [{"offset": -0.1}]})
    )
    assert catalog.boa_offset(item) == -1000


def test_newest_product_per_acquisition_is_kept_and_sorted():
    early = _item("S2A_MSIL2A_20250607T095041_R079_T33TVF_20250607T120812", T0)
    late = _item("S2A_MSIL2A_20250607T095041_R079_T33TVF_20250607T134113", T0)
    other = _item("S2B_MSIL2A_20250610T095029_R079_T33TVF_20250610T120000", T1)
    kept = catalog.newest_per_acquisition([other, late, early])
    assert [i.id for i in kept] == [late.id, other.id]
