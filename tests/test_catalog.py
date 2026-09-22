"""Pure-function checks on catalog.py: the offset rule. No network."""

from datetime import UTC, datetime

import pystac

from burnsev import catalog

T0 = datetime(2025, 6, 7, 9, 50, 41, tzinfo=UTC)


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

