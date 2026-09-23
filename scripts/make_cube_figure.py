"""Draw docs/figures/cube.png from the cached cube: one band as a 3-D stack of dated images, and
the seven bands as smaller stacks.

Run after step 3 of the notebook has built data/cache/s2_dn_20m.nc:
    python scripts/make_cube_figure.py
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D

from burnsev import aoi, catalog, indices, ingest

STACK_DATES = ["2025-06-05", "2025-07-25", "2025-08-14", "2025-09-18", "2025-10-13"]
SMALL_DATES = ["2025-06-05", "2025-08-14"]
BANDS = [
    ("B02", "blue", "pictures"),
    ("B03", "green", "pictures"),
    ("B04", "red", "pictures"),
    ("B8A", "near infrared", "burn index"),
    ("B11", "short-wave IR 1", "bonus B only"),
    ("B12", "short-wave IR 2", "burn index"),
    ("SCL", "scene classes", "cloud mask"),
]
SCL_COLOURS = ListedColormap(
    ["black", "red", "#404040", "#8b5a2b", "#2e8b57", "#e6c700", "#1f77b4", "#a0a0a0",
     "#d0d0d0", "#ffffff", "#40e0d0", "#ff69b4"]
)
ASPECT = 560 / 637  # rows over columns
FLATTEN = 0.45  # how much a layer is squashed to look like it lies flat
MONTH = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TITLE, LABEL, SMALL = 17, 13, 12  # font sizes


def grey(values: np.ndarray) -> np.ndarray:
    lo, hi = np.nanpercentile(values, [2, 98])
    scaled = (values - lo) / (hi - lo)
    return np.where(np.isnan(scaled), 0.9, np.clip(scaled, 0, 1))


def layer_transform(x0: float, y0: float, width: float, shear: float) -> Affine2D:
    """Map image coordinates (0..width, 0..height) onto a flat-lying parallelogram at (x0, y0)."""
    height = width * ASPECT
    return Affine2D().scale(1, FLATTEN).skew(np.arctan(shear / (height * FLATTEN)), 0).translate(x0, y0)


def draw_layer(ax, image, x0, y0, width, shear, cmap="gray", vmin=0, vmax=1, edge="#5b6b7c"):
    height = width * ASPECT
    trans = layer_transform(x0, y0, width, shear)
    art = ax.imshow(image, extent=[0, width, 0, height], cmap=cmap, vmin=vmin, vmax=vmax,
                    interpolation="bilinear", origin="upper")
    art.set_transform(trans + ax.transData)
    corners = trans.transform([[0, 0], [width, 0], [width, height], [0, height]])
    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor=edge, linewidth=1.2))
    return trans


def pixel_xy(trans: Affine2D, col: int, row: int, width: float) -> tuple[float, float]:
    height = width * ASPECT
    x, y = trans.transform([col / 637 * width, (1 - row / 560) * height])
    return float(x), float(y)


def main() -> Path:
    dn = ingest.load_cached(ingest.cache_path("s2_dn_20m"))
    items = catalog.search_scenes(aoi.BBOX, *aoi.PRE_WINDOW, aoi.MAX_CLOUD) + catalog.search_scenes(
        aoi.BBOX, *aoi.POST_WINDOW, aoi.MAX_CLOUD
    )
    refl = ingest.mask_and_scale(dn, catalog.offsets_by_day(items))
    dnbr = indices.dnbr(indices.nbr(refl), aoi.PRE_WINDOW, aoi.POST_WINDOW)
    row, col = np.unravel_index(int(np.nanargmax(dnbr.values)), dnbr.shape)  # a burned pixel

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.set_axis_off()
    ax.text(3.6, 7.55, "One band (B12): 46 dated images, 5 shown", ha="center", fontsize=TITLE,
            weight="bold")
    ax.text(11.2, 7.55, "Seven such stacks, one per band", ha="center", fontsize=TITLE, weight="bold")

    # Left: the big stack, oldest at the bottom, one pixel followed through it.
    width, shear, step, x0, y0 = 3.8, 1.5, 0.95, 1.3, 1.0
    marks = []
    for k, day in enumerate(STACK_DATES):
        trans = draw_layer(ax, grey(refl["B12"].sel(time=day).squeeze("time").values),
                           x0, y0 + k * step, width, shear)
        ax.text(x0 - 0.15, y0 + k * step + 0.08, f"{int(day[8:])} {MONTH[int(day[5:7]) - 1]}",
                ha="right", va="bottom", fontsize=LABEL)
        marks.append(pixel_xy(trans, col, row, width))
    (xb, yb), (xt, yt) = marks[0], marks[-1]
    ax.plot([xb, xt], [yb, yt], linestyle="--", color="#c0392b", linewidth=2)
    ax.plot([xb, xt], [yb, yt], "s", color="#c0392b", markersize=8)
    ax.text(x0 + width + shear + 0.15, yb, "red: the same\npixel on every date", color="#c0392b",
            fontsize=LABEL, ha="left", va="center")
    ax.text(3.6, 0.5, "Follow the red pixel down the stack: 46 numbers.", ha="center", fontsize=SMALL)
    ax.text(3.6, 0.2, "The median is the middle one. Each image: 637 by 560 pixels of 20 m.",
            ha="center", fontsize=SMALL, color="#555")

    # Right: a small stack per band, the same two dates in each.
    swidth, sshear, sstep = 1.35, 0.5, 0.34
    positions = [(7.7, 5.0), (9.5, 5.0), (11.3, 5.0), (13.1, 5.0), (8.6, 2.0), (10.4, 2.0), (12.2, 2.0)]
    for (band, meaning, job), (sx, sy) in zip(BANDS, positions):
        for k, day in enumerate(SMALL_DATES):
            if band == "SCL":
                image = dn["SCL"].sel(time=day).squeeze("time").values
                draw_layer(ax, image, sx, sy + k * sstep, swidth, sshear, cmap=SCL_COLOURS, vmin=0, vmax=11)
            else:
                draw_layer(ax, grey(refl[band].sel(time=day).squeeze("time").values),
                           sx, sy + k * sstep, swidth, sshear)
        cx = sx + (swidth + sshear) / 2
        ax.text(cx, sy - 0.32, band, ha="center", fontsize=LABEL + 1, weight="bold")
        ax.text(cx, sy - 0.62, meaning, ha="center", fontsize=SMALL)
        ax.text(cx, sy - 0.9, job, ha="center", fontsize=SMALL, color="#666")
    ax.text(11.2, 0.5, "NBR uses B8A and B12. Pictures use B02, B03, B04. SCL marks cloud, shadow, missing.",
            ha="center", fontsize=SMALL)
    ax.text(11.2, 0.2, "Bonus B wants all six reflectance bands, so B11 is downloaded too.",
            ha="center", fontsize=SMALL, color="#555")

    out = Path("docs/figures/cube.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white")
    return out


if __name__ == "__main__":
    print("written", main())
