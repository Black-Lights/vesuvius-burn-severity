"""Writing and checking the output files on tiny rasters. No network."""

import geopandas as gpd
import numpy as np
import rasterio
import rioxarray  # noqa: F401
import xarray as xr
from rio_cogeo.cogeo import cog_validate

from burnsev import export


def _utm(values, pixel_m=20.0):
    values = np.asarray(values)
    n_y, n_x = values.shape
    x = 450_000 + pixel_m / 2 + pixel_m * np.arange(n_x)
    y = 4_520_000 - pixel_m / 2 - pixel_m * np.arange(n_y)
    return xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs("EPSG:32633")


def test_class_raster_is_a_valid_cog_with_its_colours(tmp_path):
    classes = _utm(np.array([[0, 1], [2, 255]], dtype="uint8"))
    path = tmp_path / "classes.tif"
    export.write_cog(classes, path, nodata=255, colormap=export.colormap_from_hex(["#000000", "#ff0000", "#00ff00"]))
    assert cog_validate(str(path), quiet=True)[0]
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 32633
        assert src.read(1).tolist() == [[0, 1], [2, 255]]
        assert src.colormap(1)[1] == (255, 0, 0, 255)


def test_float_raster_keeps_values_and_writes_nodata_for_nan(tmp_path):
    path = tmp_path / "dnbr.tif"
    export.write_cog(_utm(np.array([[0.5, np.nan]], dtype="float32")), path, nodata=-9999.0)
    with rasterio.open(path) as src:
        assert src.read(1).tolist() == [[0.5, -9999.0]]


def test_perimeter_area_and_geojson_in_longitude_latitude(tmp_path):
    fire = _utm(np.array([[1, 1, 0], [1, 0, 0]], dtype="uint8"))
    perimeter = export.main_fire_perimeter(fire)
    assert perimeter["area_ha"].iloc[0] == 0.1  # three 20 m pixels = 0.12 ha, rounded to 0.1
    export.write_geojson(perimeter, tmp_path / "p.geojson")
    assert gpd.read_file(tmp_path / "p.geojson").crs.to_epsg() == 4326


def test_check_outputs_reads_only_the_files_it_is_given(tmp_path):
    export.write_cog(_utm(np.array([[0.5, 0.2]], dtype="float32")), tmp_path / "new.tif", nodata=-9999.0)
    export.write_cog(_utm(np.array([[1.0]], dtype="float32")), tmp_path / "old.tif", nodata=-9999.0)
    table = export.check_outputs([tmp_path / "new.tif"])  # old.tif, left by an earlier run, stays out
    assert table.index.tolist() == ["new.tif"]
    assert table.loc["new.tif", "check"] == "valid COG"
    assert table.loc["new.tif", "content"] == "2 x 1 px, float32"
