"""Load real pixels for a list of STAC items into one xarray cube, then mask and scale.

``odc.stac.load`` does the work: for each item and band it opens the cloud-optimised
GeoTIFF, reads only the window that covers the bounding box (an HTTP range request per
tile), reprojects to the target CRS and resolution, and stacks everything into a
Dataset with dimensions (time, y, x) and one variable per band.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import odc.stac
import planetary_computer
import pystac
import xarray as xr

from . import aoi


def load_cube(
    items: list[pystac.Item],
    bbox: tuple[float, float, float, float] = aoi.BBOX,
    bands: list[str] = aoi.BANDS,
    resolution: int = aoi.RESOLUTION,
    crs: str = aoi.CRS,
) -> xr.Dataset:
    """Download the pixels of ``bands`` for every item over ``bbox``.

    Scenes acquired on the same day (two orbits, or two granules of one orbit) are
    merged into one time step with ``groupby="solar_day"``. The result is eager
    (no dask): the AOI is small enough to hold in memory, which keeps the notebook
    free of scheduler details to reason about.
    """
    ds = odc.stac.load(
        items,
        bands=bands,
        bbox=bbox,
        crs=crs,
        resolution=resolution,
        groupby="solar_day",
        patch_url=planetary_computer.sign,
        chunks=None,
        dtype="uint16",
        nodata=0,
    )
    return ds


def mask_and_scale(
    ds: xr.Dataset,
    offsets: dict[str, int],
    scl_mask: tuple[int, ...] = aoi.SCL_MASK,
) -> xr.Dataset:
    """Turn digital numbers into surface reflectance and blank out unusable pixels.

    ``offsets`` maps each time step's date (ISO string) to the BOA offset in DN
    (-1000 for baseline >= 04.00, else 0). Reflectance = (DN + offset) / 10000.
    Pixels whose SCL class is in ``scl_mask`` (no data, saturated, shadow, cloud,
    cirrus, snow) become NaN in every reflectance band, so that later medians and
    index calculations ignore them instead of averaging cloud into the signal.
    """
    scl = ds["SCL"]
    bad = xr.zeros_like(scl, dtype=bool)
    for cls in scl_mask:
        bad = bad | (scl == cls)

    dates = [str(np.datetime64(t, "D")) for t in ds["time"].values]
    offset = xr.DataArray(
        [offsets[d] for d in dates], dims=["time"], coords={"time": ds["time"]}
    )

    out = xr.Dataset(coords=ds.coords)
    for name in ds.data_vars:
        if name == "SCL":
            continue
        refl = (ds[name].astype("float32") + offset.astype("float32")) / 10000.0
        refl = refl.where(~bad)
        # Reflectance outside [0, 1] is physically impossible: clamp small negatives
        # from the offset and drop the rest.
        refl = refl.where(refl <= 1.0).clip(min=0.0)
        out[name] = refl.astype("float32")
    out["valid_fraction"] = (~bad).mean(dim=("y", "x")).astype("float32")
    out.attrs["scl_mask"] = list(scl_mask)
    return out


def cache_path(name: str, cache_dir: Path = Path("data/cache")) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name}.nc"


def save_cube(ds: xr.Dataset, path: Path) -> None:
    """Persist a cube so re-running the notebook does not re-download 12 scenes."""
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, engine="h5netcdf", encoding=encoding)


def load_cached(path: Path) -> xr.Dataset | None:
    if path.exists():
        return xr.open_dataset(path, engine="h5netcdf").load()
    return None
