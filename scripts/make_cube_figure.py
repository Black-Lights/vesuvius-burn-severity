"""Draw docs/figures/cube.png from the cached cube: one band as a stack of dated images, and
the seven bands of one date side by side.

Run after step 3 of the notebook has built data/cache/s2_dn_20m.nc:
    python scripts/make_cube_figure.py
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from burnsev import aoi, catalog, indices, ingest

STACK_DATES = ["2025-06-05", "2025-06-07", "2025-06-15", "2025-07-25", "2025-10-08", "2025-10-13"]
SHOW_DATE = "2025-08-14"
BANDS = [
    ("B02", "blue", "pictures"),
    ("B03", "green", "pictures"),
    ("B04", "red", "pictures"),
    ("B8A", "near infrared", "burn index"),
    ("B11", "short-wave IR 1", "bonus B model only"),
    ("B12", "short-wave IR 2", "burn index"),
    ("SCL", "scene classes", "cloud mask"),
]
SCL_COLOURS = ListedColormap(
    ["black", "red", "#404040", "#8b5a2b", "#2e8b57", "#e6c700", "#1f77b4", "#a0a0a0",
     "#d0d0d0", "#ffffff", "#40e0d0", "#ff69b4"]
)


def grey(values: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(values, [2, 98])
    scaled = (values - lo) / (hi - lo)
    return np.where(np.isnan(scaled), 0.9, np.clip(scaled, 0, 1))


def main() -> Path:
    dn = ingest.load_cached(ingest.cache_path("s2_dn_20m"))
    items = catalog.search_scenes(aoi.BBOX, *aoi.PRE_WINDOW, aoi.MAX_CLOUD) + catalog.search_scenes(
        aoi.BBOX, *aoi.POST_WINDOW, aoi.MAX_CLOUD
    )
    refl = ingest.mask_and_scale(dn, catalog.offsets_by_day(items))
    nbr = indices.nbr(refl)
    dnbr = indices.dnbr(nbr, aoi.PRE_WINDOW, aoi.POST_WINDOW)
    row, col = np.unravel_index(int(np.nanargmax(dnbr.values)), dnbr.shape)  # a burned pixel

    fig = plt.figure(figsize=(14, 7.2))
    fig.text(0.21, 0.95, "One band (B12): 46 dated images in a stack, 6 shown",
             ha="center", fontsize=14, weight="bold")
    fig.text(0.71, 0.95, f"Seven such stacks, one per band ({SHOW_DATE} shown)",
             ha="center", fontsize=14, weight="bold")

    # Left: a cascade of real B12 images, oldest at the bottom, one pixel followed through them.
    w, h = 0.24, 0.24 * (560 / 637) * (14 / 7.2)
    corners = []
    for k, day in enumerate(STACK_DATES):
        left, bottom = 0.04 + 0.024 * k, 0.10 + 0.052 * k
        ax = fig.add_axes([left, bottom, w, h])
        ax.imshow(grey(refl["B12"].sel(time=day).squeeze("time").values), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#5b6b7c")
        month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        fig.text(0.036, bottom + 0.012, f"{int(day[8:])} {month[int(day[5:7]) - 1]}",
                 ha="right", va="bottom", fontsize=10)
        corners.append((left + w * col / 637, bottom + h * (1 - row / 560)))
        if k in (0, len(STACK_DATES) - 1):
            ax.plot(col, row, "s", color="#c0392b", markersize=7)
    (x0, y0), (x1, y1) = corners[0], corners[-1]
    fig.add_artist(plt.Line2D([x0, x1], [y0, y1], color="#c0392b", linestyle="--", linewidth=1.5))
    fig.text(0.21, 0.80, "red square: the same pixel on every date", color="#c0392b", fontsize=10,
             ha="center")
    fig.text(0.21, 0.035, "Follow the red pixel down the stack: 46 numbers. The median is the middle one.\n"
             "Each image: 637 by 560 pixels of 20 m.", ha="center", fontsize=10, color="#333")

    # Right: the seven bands of one date.
    tw, th = 0.105, 0.105 * (560 / 637) * (14 / 7.2)
    positions = [(0.47, 0.60), (0.60, 0.60), (0.73, 0.60), (0.86, 0.60), (0.535, 0.20), (0.665, 0.20), (0.795, 0.20)]
    day = refl.sel(time=SHOW_DATE).squeeze("time")
    scl = dn["SCL"].sel(time=SHOW_DATE).squeeze("time").values
    for (band, meaning, job), (left, bottom) in zip(BANDS, positions):
        ax = fig.add_axes([left, bottom, tw, th])
        if band == "SCL":
            ax.imshow(scl, cmap=SCL_COLOURS, vmin=0, vmax=11, interpolation="nearest")
        else:
            ax.imshow(grey(day[band].values), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.text(left + tw / 2, bottom - 0.035, band, ha="center", fontsize=11, weight="bold")
        fig.text(left + tw / 2, bottom - 0.065, meaning, ha="center", fontsize=10)
        fig.text(left + tw / 2, bottom - 0.095, job, ha="center", fontsize=10, color="#666")
    fig.text(0.71, 0.035, "NBR uses B8A and B12. Pictures use B02, B03, B04. SCL marks cloud, shadow and missing pixels.\n"
             "Bonus B wants all six reflectance bands, so B11 is downloaded too.",
             ha="center", fontsize=10, color="#333")

    out = Path("docs/figures/cube.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white")
    return out


if __name__ == "__main__":
    print("written", main())
