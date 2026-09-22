"""Load real pixels for a list of STAC items into one xarray cube, then mask and scale.

``odc.stac.load`` does the work: for each item and band it opens the cloud-optimised
GeoTIFF, reads only the window that covers the bounding box (an HTTP range request per
internal tile), reprojects to the target CRS and resolution, and stacks everything into a
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
    threads: int = 8,
) -> xr.Dataset:
    """Download the pixels of ``bands`` for every item over ``bbox``, as digital numbers.

    One time step per solar day: products from the same day (two orbits ten minutes apart,
    or one acquisition published twice) are fused, first valid pixel wins. The result is
    eager, no dask, because the box is small enough to hold in memory; ``threads`` file
    reads run in parallel. Nodata is 0 in Sentinel-2 L2A files and stays 0 here.
    """
    return odc.stac.load(
        items,
        bands=bands,
        bbox=bbox,
        crs=crs,
        resolution=resolution,
        groupby="solar_day",
        patch_url=planetary_computer.sign,
        chunks=None,
        pool=threads,
        dtype="uint16",
        nodata=0,
    )


def mask_and_scale(
    ds: xr.Dataset,
    offsets: dict[str, int],
    scl_mask: tuple[int, ...] = aoi.SCL_MASK,
) -> xr.Dataset:
    """Turn digital numbers into surface reflectance and blank out unusable pixels.

    ``offsets`` maps each time step's date (ISO string) to the BOA offset in DN
    (ESA BOA_ADD_OFFSET: -1000 for baseline >= 04.00, else 0). Then
    reflectance = (DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE, the latter being 10000.
    Pixels whose SCL class is in ``scl_mask`` (no data, saturated, shadow, cloud,
    cirrus, snow) become NaN in every band, so later medians and indices ignore them
    instead of averaging cloud into the signal. ``valid_fraction`` records, per date,
    the share of pixels that survived, for the quality report.
    """
    bad = ds["SCL"].isin(list(scl_mask))
    dates = [str(np.datetime64(t, "D")) for t in ds["time"].values]
    offset = xr.DataArray(
        [offsets[d] for d in dates], dims=["time"], coords={"time": ds["time"]}
    )

    out = xr.Dataset(coords=ds.coords)
    for name in ds.data_vars:
        if name == "SCL":
            continue
        dn_shifted = ds[name].astype("float32") + offset.astype("float32")
        refl = dn_shifted / float(aoi.BOA_QUANTIFICATION_VALUE)
        refl = refl.where(~bad)
        # Reflectance outside [0, 1] is physically impossible: clamp the small negatives
        # left by the offset to 0 and drop anything above 1.
        out[name] = refl.where(refl <= 1.0).clip(min=0.0).astype("float32")
    out["valid_fraction"] = (~bad).mean(dim=("y", "x")).astype("float32")
    out.attrs["scl_mask"] = list(scl_mask)
    return out


def cache_path(name: str, cache_dir: Path = Path("data/cache")) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name}.nc"


def save_cube(ds: xr.Dataset, path: Path) -> None:
    """Store the downloaded cube (digital numbers and SCL) so a rerun skips the download.

    Only the download is cached. Masking and scaling are recomputed on every run, so the
    cache can never hide a change in the processing.
    """
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(path, engine="netcdf4", encoding=encoding)


def load_cached(path: Path) -> xr.Dataset | None:
    """Read a cube written by ``save_cube``; None if the file is not there.

    NetCDF stores the CRS as a variable called ``spatial_ref``; xarray reads it back as data
    rather than as a coordinate, so it is moved back, otherwise it would be treated as a band.
    """
    if not path.exists():
        return None
    ds = xr.open_dataset(path, engine="netcdf4").load()
    if "spatial_ref" in ds.data_vars:
        ds = ds.set_coords("spatial_ref")
    return ds
