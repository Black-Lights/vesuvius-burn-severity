"""Bonus B, second model: TerraMind land cover, what the burned ground was before the fire.

TerraMind 1.0 base (IBM, ESA and Forschungszentrum Julich) is a generative model pretrained on nine
million places seen by several sensors at once (the TerraMesh dataset). From a Sentinel-2 L2A image
it generates a land cover map in the classes of ESRI's yearly 10 m map, which it learned from. dNBR
and NDVI say how much the vegetation changed; they cannot say whether it was forest, shrub, fields
or houses.

The model card calls a generation a "mental image": a sample, not a measurement. So the map is the
majority of several samples with fixed seeds, the number of samples behind each pixel is kept, and
the map is checked against ESRI's own map of 2023.

torch and terratorch are imported inside the functions that use them, as in prithvi.py. Without
them, the notebook reads the maps saved by the last run.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import odc.stac
import pandas as pd
import planetary_computer
import rasterio
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.xr import wrap_xr
from scipy import ndimage

from . import aoi, catalog, decision, export, ingest, prithvi

MODEL = "terramind_v1_base_generate"  # its name in TerraTorch; weights from Hugging Face, Apache 2.0
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]  # model order
CLASSES = ["no data", "water", "trees", "flooded vegetation", "crops", "built area", "bare ground",
           "snow and ice", "clouds", "rangeland"]  # the model's output order
ESRI_CODES = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11]  # the same classes, as numbered in ESRI's own files
ESRI_COLLECTION = "io-lulc-annual-v02"  # ESRI's yearly map on Planetary Computer, 2017 to 2023
PIXEL_M = 10  # the model's pixel size
TILE = 224  # the model reads 224 x 224 pixels at a time
SEEDS = [0, 1, 2, 3, 4]  # one sample per seed; the map is their majority
BUILT = CLASSES.index("built area")
TREES = CLASSES.index("trees")


def grid_10m(like: xr.DataArray) -> GeoBox:
    """The 10 m grid over the box of the 20 m grid ``like``: each 20 m pixel is 2 x 2 of these.
    ``tight`` keeps the corner where it is instead of snapping it to a multiple of 10 m."""
    return GeoBox.from_bbox(like.odc.geobox.boundingbox, like.odc.geobox.crs, tight=True, resolution=PIXEL_M)


def to_10m(values: xr.DataArray) -> np.ndarray:
    """A 20 m array on the 10 m grid: each 20 m pixel becomes its own 2 x 2 pixels, exactly."""
    return values.values.repeat(2, axis=0).repeat(2, axis=1)


def load_bands(day: str, like: xr.DataArray) -> np.ndarray:
    """The 12 bands of one date on the 10 m grid, as the model was trained: (12, rows, cols).

    TerraMesh stores reflectance x 10000 with the +1000 offset of recent products removed, and
    brings the 20 and 60 m bands to 10 m by nearest; the same is done here. Downloaded once,
    then read from data/cache.
    """
    items = catalog.search_scenes(aoi.BBOX, day, day, aoi.MAX_CLOUD)
    path = ingest.cache_path(f"s2_dn_10m_{day.replace('-', '')}")
    dn = ingest.load_cached(path)
    if dn is None:
        dn = odc.stac.load(items, bands=[*BANDS, "SCL"], geobox=grid_10m(like), groupby="solar_day",
                           patch_url=planetary_computer.sign, chunks=None, dtype="uint16", nodata=0,
                           resampling="nearest")
        ingest.save_cube(dn, path)
    refl = ingest.mask_and_scale(dn, catalog.offsets_by_day(items))
    return refl[BANDS].isel(time=0).to_array().values * 10_000


def load_esri(like: xr.DataArray, year: int = 2023) -> np.ndarray:
    """ESRI's own land cover map of ``year`` on the 10 m grid, in the model's class numbers."""
    search = catalog.open_catalog().search(collections=[ESRI_COLLECTION], bbox=list(aoi.BBOX))
    items = [item for item in search.items() if item.id.endswith(f"-{year}")]  # ids like 33T-2023
    codes = odc.stac.load(items, bands=["data"], geobox=grid_10m(like), patch_url=planetary_computer.sign,
                          chunks=None, dtype="uint8", nodata=0, resampling="nearest")["data"].isel(time=0)
    return esri_to_classes(codes.values)


def esri_to_classes(codes: np.ndarray) -> np.ndarray:
    """ESRI's codes (1 water, 2 trees, 4 flooded vegetation, ...) as the model's class numbers."""
    to_class = np.zeros(256, dtype="uint8")  # a lookup table: ESRI code -> class number
    to_class[ESRI_CODES] = np.arange(len(ESRI_CODES))
    return to_class[codes]


def load_model(device: str):
    """TerraMind 1.0 base, set up to generate land cover from Sentinel-2 L2A (1.5 GB the first
    time). ``standardize`` applies the model's own training means and standard deviations. The
    land cover decoder (a VQ-VAE) works in one pass, so ``timesteps`` (10, as in the model card's
    example) changes nothing here: it sets the steps of the diffusion decoders of image outputs."""
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)  # see prithvi.load_model
    from terratorch import FULL_MODEL_REGISTRY

    root_handlers = logging.getLogger().handlers[:]  # see prithvi.load_model
    model = FULL_MODEL_REGISTRY.build(MODEL, pretrained=True, modalities=["S2L2A"],
                                      output_modalities=["LULC"], timesteps=10, standardize=True)
    logging.getLogger().handlers = root_handlers
    return model.eval().to(device)


def generate(model, image: np.ndarray, device: str, seed: int) -> np.ndarray:
    """One land cover sample for the whole image: (rows, cols) of class numbers.

    The image is mirrored out to whole tiles of 224 x 224, each row of tiles goes through the model
    as one batch, and each pixel takes the class with the highest score. ``seed`` fixes the random
    choices of the generation, so the same seed gives the same map.
    """
    import torch

    random.seed(seed)  # TerraTorch draws its sampling seed from Python's random module
    torch.manual_seed(seed)  # TerraTorch reseeds torch from that draw; kept in case a version does not
    rows, cols = image.shape[1:]
    padded = np.pad(np.nan_to_num(image), ((0, 0), (0, -rows % TILE), (0, -cols % TILE)), mode="reflect")
    classes = np.zeros(padded.shape[1:], dtype="uint8")
    for top in range(0, padded.shape[1], TILE):
        lefts = range(0, padded.shape[2], TILE)
        batch = np.stack([padded[:, top : top + TILE, left : left + TILE] for left in lefts])
        with torch.no_grad():
            scores = model(torch.from_numpy(batch.astype("float32")).to(device))["LULC"]  # (tiles, 10, 224, 224)
        labels = scores.argmax(dim=1).cpu().numpy()
        for k, left in enumerate(lefts):
            classes[top : top + TILE, left : left + TILE] = labels[k]
    return classes[:rows, :cols]


def majority(samples: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """The class most samples give each pixel, and how many samples gave it. A tie goes to the
    lower class number."""
    stack = np.stack(samples)
    votes = np.stack([(stack == c).sum(axis=0) for c in range(len(CLASSES))])  # (classes, rows, cols)
    return votes.argmax(axis=0).astype("uint8"), votes.max(axis=0).astype("uint8")


def save(values: np.ndarray, like: xr.DataArray, path: Path, colours: list[str] | None, tags: dict) -> None:
    """Write a 10 m class raster as a COG, with a colour table when ``colours`` is given."""
    colormap = export.colormap_from_hex(colours) if colours else None
    export.write_cog(wrap_xr(values, grid_10m(like)), path, nodata=0, colormap=colormap, tags=tags)


def read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def land_cover(day: str, like: xr.DataArray, out_dir: Path, colours: list[str]) -> tuple[np.ndarray, np.ndarray, str]:
    """The land cover of ``day`` on the 10 m grid, the votes behind each pixel, and a line saying
    where they came from. Computed and saved with torch and terratorch installed; otherwise read
    from the files saved by the last run."""
    tag = day.replace("-", "")
    cover_file = out_dir / f"terramind_land_cover_{tag}_10m.tif"
    votes_file = out_dir / f"terramind_votes_{tag}_10m.tif"
    if not prithvi.installed():
        return read(cover_file), read(votes_file), "read from the saved maps (torch not installed)"
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    image = load_bands(day, like)
    cover, votes = majority([generate(model, image, device, seed) for seed in SEEDS])
    source = "TerraMind 1.0 base, Sentinel-2 L2A " + day
    save(cover, like, cover_file, colours, {"classes": ", ".join(f"{i} {n}" for i, n in enumerate(CLASSES)),
                                            "source": f"{source}, majority of {len(SEEDS)} samples"})
    save(votes, like, votes_file, None, {"quantity": f"samples of {len(SEEDS)} giving the majority class",
                                         "source": source})
    return cover, votes, f"computed on {device}, {len(SEEDS)} samples"


def hectares(mask: np.ndarray) -> float:
    return round(float(mask.sum()) * PIXEL_M**2 / 10_000, 1)


def check_against_esri(cover: np.ndarray, esri: np.ndarray) -> pd.DataFrame:
    """Per class: hectares in each map, and the share of ESRI's pixels the model gives the same class."""
    rows = {}
    for code, name in enumerate(CLASSES):
        in_esri, in_model = esri == code, cover == code
        if in_esri.any() or in_model.any():
            rows[name] = {"ESRI 2023 (ha)": hectares(in_esri), "TerraMind (ha)": hectares(in_model),
                          "ESRI pixels matched (%)": round(100 * float(in_model[in_esri].mean()), 1) if in_esri.any() else None}
    return pd.DataFrame(rows).T.rename_axis("class")


def cover_by_severity(cover: np.ndarray, severity: xr.DataArray, fire: xr.DataArray,
                      severe_steep: xr.DataArray, names: dict[int, str]) -> pd.DataFrame:
    """Hectares of each land cover in each severity class of the main fire, and on the severe and
    steep ground that the priority rule counts."""
    severity_10m, fire_10m, steep_10m = to_10m(severity), to_10m(fire), to_10m(severe_steep)
    columns = {}
    for code, name in names.items():
        in_class = fire_10m & (severity_10m == code)
        if in_class.any():
            columns[name] = {CLASSES[c]: hectares(in_class & (cover == c)) for c in range(len(CLASSES))}
    columns["severe and steep"] = {CLASSES[c]: hectares(steep_10m & (cover == c)) for c in range(len(CLASSES))}
    table = pd.DataFrame(columns).rename_axis("land cover before the fire")
    return table[table.sum(axis=1) > 0]


def cell_exposure(cover: np.ndarray, fire: xr.DataArray, like: xr.DataArray,
                  cell_m: float = aoi.GRID_CELL_M) -> pd.DataFrame:
    """For each grid cell: the share of its burned ground that was trees before the fire, and the
    straight-line distance from the cell to the nearest built area, in metres.

    Distance is not a flow path: a cell above a village on the other side of a ridge is near it,
    and its debris would not reach it.
    """
    grid = wrap_xr(cover, grid_10m(like))
    rows, cols = decision.grid_index(grid, cell_m)
    # For every pixel, the distance to the nearest built pixel (0 on built ground).
    distance = ndimage.distance_transform_edt(cover != BUILT) * PIXEL_M
    burned = to_10m(fire)
    pixels = pd.DataFrame({"row": rows.ravel(), "col": cols.ravel(), "distance": distance.ravel(),
                           "trees": (burned & (cover == TREES)).ravel(), "burned": burned.ravel()})
    cells = pixels.groupby(["row", "col"]).agg(nearest_built_m=("distance", "min"), trees=("trees", "sum"),
                                               burned=("burned", "sum"))
    cells["trees_pct"] = (100 * cells["trees"] / cells["burned"]).round(0)
    return cells[cells["burned"] > 0][["trees_pct", "nearest_built_m"]].reset_index()
