"""Bonus B, first model: Prithvi-EO-2.0 burn scars, a second burn map from one image after the fire.

The model (IBM and NASA) is a 300-million-parameter vision transformer, fine-tuned on about 800
chips of 512 x 512 pixels from fires in the United States, at 30 m (HLS). From six bands of one
date it labels each pixel burned or not burned. It needs no pre-fire image and no threshold of
ours, so it checks the dNBR burn map independently. It does not replace it: the priority rule
needs the severity classes, and the model only says burned or not.

torch and terratorch are imported inside the functions that use them, so the rest of the package
works without them (they are in requirements-gfm.txt). Without them, the notebook reads the maps
saved by the last run.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr

from . import aoi, export, indices

REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars"  # on Hugging Face, Apache 2.0
CONFIG_FILE = "burn_scars_config.yaml"
WEIGHTS_FILE = "Prithvi_EO_V2_300M_BurnScars.pt"  # 1.3 GB, downloaded once to the Hugging Face cache
BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]  # blue, green, red, narrow NIR, SWIR 1, SWIR 2
MODEL_PIXEL_M = 30  # the training images (HLS) have 30 m pixels
CHIP = 512  # the model reads 512 x 512 pixels at a time
NO_DATA = 255  # in the saved files, which store the probability in percent


def installed() -> bool:
    """True when torch and terratorch are installed, so the model can run here."""
    return importlib.util.find_spec("terratorch") is not None


def to_30m(refl: xr.Dataset, day: str) -> np.ndarray:
    """The six bands of one date on a 30 m grid, as an array (6, rows, cols) of reflectance.

    Each 20 m pixel is cut into four 10 m pixels with the same value, then every block of 3 x 3
    of those is averaged into one 30 m pixel. That is an area-weighted average, the method HLS
    uses to bring Sentinel-2's 20 m bands to 30 m, so the input is built like the training
    images. The 30 m grid starts at the corner of the 20 m grid; the last 10 or 20 m that do not
    fill a 30 m pixel are dropped. A 30 m pixel that touches a masked pixel stays NaN.
    """
    image = refl[BANDS].sel(time=day).squeeze("time").to_array().values  # (6, rows, cols), 20 m
    at_10m = image.repeat(2, axis=1).repeat(2, axis=2)
    rows, cols = at_10m.shape[1] // 3, at_10m.shape[2] // 3
    blocks = at_10m[:, : rows * 3, : cols * 3].reshape(len(BANDS), rows, 3, cols, 3)
    return blocks.mean(axis=(2, 4))


def make_chip(image: np.ndarray, means: list[float], stds: list[float]) -> np.ndarray:
    """Fill the image out to one 512 x 512 chip and standardise it, as in training.

    Missing values become 0, as in the model's config (no_data_replace: 0). The chip is filled
    by mirroring the image at its bottom and right edges, as the model's own inference script
    does, so the model sees land rather than an empty border. Each band is then standardised
    with the mean and standard deviation of the training images.
    """
    rows, cols = image.shape[1:]
    filled = np.nan_to_num(image, nan=0.0)
    chip = np.pad(filled, ((0, 0), (0, CHIP - rows), (0, CHIP - cols)), mode="reflect")
    mean = np.array(means).reshape(-1, 1, 1)
    std = np.array(stds).reshape(-1, 1, 1)
    return ((chip - mean) / std).astype("float32")


def load_model(device: str):
    """Build the model from its config, load the fine-tuned weights, and return the model with
    the training means and standard deviations of the six bands."""
    # torch notes at import that triton, a GPU compiler it does not need here, is missing.
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
    import torch
    import yaml
    from huggingface_hub import hf_hub_download
    from terratorch.tasks import SemanticSegmentationTask

    with open(hf_hub_download(REPO, CONFIG_FILE), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    task_args = config["model"]["init_args"]
    # The general Prithvi weights are not needed: the fine-tuned weights below replace every layer.
    task_args["model_args"]["backbone_pretrained"] = False
    # Building the model calls logging.debug() on the root logger, which in Python switches on
    # log output for every library (logging.basicConfig). Put the root logger back as it was.
    root_handlers = logging.getLogger().handlers[:]
    task = SemanticSegmentationTask(**task_args)
    logging.getLogger().handlers = root_handlers
    # weights_only: read tensors only, never run code stored in the file.
    checkpoint = torch.load(hf_hub_download(REPO, WEIGHTS_FILE), map_location="cpu", weights_only=True)
    task.load_state_dict(checkpoint["state_dict"])  # strict: fails if any layer does not match
    data = config["data"]["init_args"]
    return task.model.eval().to(device), data["means"], data["stds"]


def predict(model, chip: np.ndarray, device: str) -> np.ndarray:
    """Probability that each pixel of the chip is a burn scar, (512, 512) from 0 to 1."""
    import torch

    x = torch.from_numpy(chip).unsqueeze(0).to(device)  # (1, 6, 512, 512): a batch of one chip
    with torch.no_grad():  # inference only: no gradients, less memory
        scores = model(x).output  # (1, 2, 512, 512): one score for "not burned", one for "burned"
    return torch.softmax(scores, dim=1)[0, 1].cpu().numpy()


def burn_probability(refl: xr.Dataset, day: str, model, means, stds, device: str) -> np.ndarray:
    """to_30m, make_chip and predict for one date, cut back to the image: (rows, cols) at 30 m,
    NaN where a band was missing."""
    image = to_30m(refl, day)
    probability = predict(model, make_chip(image, means, stds), device)
    probability = probability[: image.shape[1], : image.shape[2]]
    return np.where(np.isnan(image).any(axis=0), np.nan, probability)


def grid_30m(values: np.ndarray, like: xr.DataArray) -> xr.DataArray:
    """A 30 m array with coordinates and CRS, on the grid that to_30m builds from ``like``."""
    left = float(like.x[0]) - aoi.RESOLUTION / 2
    top = float(like.y[0]) + aoi.RESOLUTION / 2
    x = left + MODEL_PIXEL_M * (np.arange(values.shape[1]) + 0.5)
    y = top - MODEL_PIXEL_M * (np.arange(values.shape[0]) + 0.5)
    return xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x}).rio.write_crs(like.rio.crs)


def to_20m(values: np.ndarray, like: xr.DataArray) -> xr.DataArray:
    """Back onto the notebook's 20 m grid: each 20 m pixel takes the value of the 30 m pixel
    that contains its centre. Nearest, not an average, because it carries labels too."""
    centres_y = (np.arange(like.shape[0]) + 0.5) * aoi.RESOLUTION  # metres from the top edge
    centres_x = (np.arange(like.shape[1]) + 0.5) * aoi.RESOLUTION  # metres from the left edge
    rows = np.minimum(centres_y // MODEL_PIXEL_M, values.shape[0] - 1).astype(int)
    cols = np.minimum(centres_x // MODEL_PIXEL_M, values.shape[1] - 1).astype(int)
    return like.copy(data=values[np.ix_(rows, cols)])


def save(probability: np.ndarray, like: xr.DataArray, path: Path, day: str) -> None:
    """Write the probability as a COG in percent (0 to 100, 255 no data), a few tens of KB."""
    percent = np.where(np.isnan(probability), NO_DATA, np.round(probability * 100)).astype("uint8")
    export.write_cog(grid_30m(percent, like), path, nodata=NO_DATA, overview_resampling="average",
                     tags={"quantity": f"burn scar probability (%), Prithvi-EO-2.0-300M-BurnScars, {day}",
                           "source": "Sentinel-2 L2A, Microsoft Planetary Computer, averaged to 30 m"})


def read(path: Path) -> np.ndarray:
    """A probability saved by ``save``, back to 0 to 1 with NaN for no data."""
    with rasterio.open(path) as src:
        percent = src.read(1)
    return np.where(percent == NO_DATA, np.nan, percent / 100)


def burn_maps(refl: xr.Dataset, days: list[str], like: xr.DataArray, out_dir: Path) -> tuple[dict, str]:
    """The burn probability at 30 m for each day, and a line saying where it came from.

    With torch and terratorch installed, the model runs (on the GPU if there is one) and each
    map is saved to ``out_dir``. Without them, the maps saved by the last run are read back.
    """
    files = {day: out_dir / f"prithvi_burn_probability_{day.replace('-', '')}_30m.tif" for day in days}
    if not installed():
        return {day: read(path) for day, path in files.items()}, "read from the saved maps (torch not installed)"
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, means, stds = load_model(device)
    maps = {}
    for day, path in files.items():
        maps[day] = burn_probability(refl, day, model, means, stds, device)
        save(maps[day], like, path, day)
    return maps, f"computed on {device}"


def found_by_class(burned: xr.DataArray, severity: xr.DataArray) -> dict[str, float]:
    """For each dNBR severity class, the percentage of its pixels that the model calls burned."""
    shares = {}
    for code, name in indices.class_names().items():
        in_class = severity.values == code
        if in_class.any():
            shares[name] = round(100 * float(burned.values[in_class].mean()), 1)
    return shares
