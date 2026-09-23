"""Pictures of the cube. Display only: nothing here changes a number in the analysis."""

from __future__ import annotations

import base64
import io
from typing import NamedTuple

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor used below)
import xarray as xr
from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, to_rgba
from matplotlib.figure import Figure
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "legend.fontsize": 12})



def fit_figures_to_width(dpi: int = 100, max_width: int = 1000) -> None:
    """Draw every figure at the width of the notebook's output area, up to ``max_width`` pixels.

    Notebook front ends show a PNG at its own pixel width, so a figure wider than the pane
    scrolls sideways. Each figure is shown instead as an HTML image at 100 % width, capped
    and centred so a wide browser window does not blow it up; stored once. Outside IPython
    this does nothing.
    """
    from IPython import get_ipython

    shell = get_ipython()
    if shell is None:
        return

    def as_html(fig: Figure) -> str:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
        data = base64.b64encode(buffer.getvalue()).decode("ascii")
        style = f"width:100%; max-width:{max_width}px; height:auto; display:block; margin:auto"
        return f'<img src="data:image/png;base64,{data}" style="{style}">'

    # Starting the notebook plotting backend resets every figure formatter, so start it first.
    plt.close(plt.figure())
    formatters = shell.display_formatter.formatters
    formatters["image/png"].pop(Figure, None)
    formatters["text/html"].for_type(Figure, as_html)


TRUE_COLOUR = ("B04", "B03", "B02")  # red, green, blue
SWIR_COLOUR = ("B12", "B8A", "B04")  # short-wave infrared, near infrared, red
MISSING_GREY = 0.9  # masked pixels (cloud, shadow, outside the swath) are drawn light grey


def stretch_bounds(
    ds: xr.Dataset, bands: tuple[str, ...], low: float = 2, high: float = 98
) -> dict[str, tuple[float, float]]:
    """Black and white points per band: the ``low`` and ``high`` percentiles of its values.

    Land reflectance sits mostly between 0.02 and 0.30, so drawn on the full 0 to 1 range an
    image is nearly black. Percentiles ignore the few extreme pixels (a bright roof, a deep
    shadow) that would otherwise set the range. 2 and 98 are the usual convention (the QGIS
    default). Computed once on a reference date and reused, so different dates compare.
    """
    return {
        b: tuple(float(v) for v in np.nanpercentile(ds[b].values, [low, high])) for b in bands
    }


def to_rgb(
    ds_day: xr.Dataset, bands: tuple[str, ...], bounds: dict[str, tuple[float, float]]
) -> np.ndarray:
    """Stack three bands into an (y, x, 3) image, each scaled 0 to 1 within its bounds."""
    layers = []
    for b in bands:
        lo, hi = bounds[b]
        scaled = (ds_day[b].values - lo) / (hi - lo)
        scaled = np.where(np.isnan(scaled), MISSING_GREY, np.clip(scaled, 0.0, 1.0))
        layers.append(scaled)
    return np.dstack(layers)


def usable_share(refl: xr.Dataset, orbits: dict[str, set[int]], fire: tuple[str, str]) -> Figure:
    """Share of the box that is usable on each date, in percent, coloured by the orbit that saw it.

    ``orbits`` maps each date to the relative orbits of its products (two when both orbits
    passed the same day). Fire days are shaded.
    """
    share = refl["valid_fraction"].to_pandas() * 100
    labels = {}
    for day in share.index:
        seen = orbits[str(day.date())]
        labels[day] = "both orbits" if len(seen) > 1 else f"orbit {next(iter(seen))}"
    colours = {"orbit 79": "tab:blue", "orbit 122": "tab:purple", "both orbits": "tab:green"}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(share.index, share.values, color="lightgrey", zorder=1)
    for label, colour in colours.items():
        days = [d for d in share.index if labels[d] == label]
        if days:
            ax.scatter(days, share[days], color=colour, label=label, zorder=2)
    ax.axvspan(pd.Timestamp(fire[0]), pd.Timestamp(fire[1]), color="orange", alpha=0.5, label="fire")
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=(1, 15)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_xlabel("2025")
    ax.set_ylabel("usable share of the box (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    fig.tight_layout()
    return fig


def before_after(refl: xr.Dataset, pre_day: str, post_day: str) -> Figure:
    """Two rows (true colour, short-wave infrared colour) by two dates, one stretch per band.

    The bounds come from the pre-fire date only. Stretching each date on its own would push
    the burn, a few percent of the pixels, into the dark end and make the pair incomparable.
    """
    pre = refl.sel(time=pre_day).squeeze("time")
    post = refl.sel(time=post_day).squeeze("time")
    rows = (("True colour", TRUE_COLOUR), ("Short-wave infrared colour", SWIR_COLOUR))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.6))
    for (label, bands), row in zip(rows, axes):
        bounds = stretch_bounds(pre, bands)
        for ax, (day, ds_day) in zip(row, ((pre_day, pre), (post_day, post))):
            ax.imshow(to_rgb(ds_day, bands, bounds))
            usable = float(ds_day["valid_fraction"]) * 100
            ax.set_title(f"{label}\n{day}, {usable:.0f}% usable")
            ax.set_axis_off()
    fig.tight_layout()
    return fig


SEVERITY_COLOURS = ["#eef3ea", "#ffffb2", "#fd8d3c", "#e31a1c", "#67000d"]  # unburned to high
DNBR_COLOURS = {"cmap": "RdYlGn_r", "vmin": -0.3, "vmax": 0.9}  # green unchanged, red burned
NDVI_COLOURS = {"cmap": "YlGn", "vmin": 0.0, "vmax": 0.8}  # pale yellow bare ground, dark green dense
EXCLUDED_GREY = "#8c8c8c"


def severity_map(
    severity: xr.DataArray, areas: dict[str, float], excluded: xr.DataArray | None = None
) -> Figure:
    """The class raster in the usual burn-severity colours, hectares per class in the legend.

    ``excluded`` marks burned pixels left out of the analysis (outside the main fire); they are
    drawn grey with their total, so what was dropped stays visible.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    areas = dict(areas)
    fig, ax = plt.subplots(figsize=(9, 8))
    shown = severity.where(severity != 255).values.astype(float)  # no data drawn white
    extra = []
    if excluded is not None:
        shown[excluded.values] = 5
        ha = float(excluded.sum()) * float(abs(severity.x[1] - severity.x[0])) ** 2 / 10_000
        areas["unburned"] = areas.get("unburned", 0.0) - ha  # the grey pixels get their own entry
        extra = [Patch(color=EXCLUDED_GREY, label=f"outside the main fire: {ha:,.0f} ha")]
    handles = [
        Patch(color=c, label=f"{n}: {areas[n]:,.0f} ha") for n, c in zip(areas, SEVERITY_COLOURS)
    ] + extra
    colours = ListedColormap([*SEVERITY_COLOURS, EXCLUDED_GREY])
    ax.imshow(shown, cmap=colours, vmin=0, vmax=5, interpolation="nearest")
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
              title="dNBR class, Key and Benson (2006)", frameon=False)
    ax.set_title("Burn severity class per 20 m pixel")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def _area_means(ax, index: xr.DataArray, masks, min_usable: float) -> None:
    """One line per area: the mean of ``index`` over the area on each date that is usable enough."""
    for mask, label, colour in masks:
        sub = index.where(mask)
        usable = sub.notnull().sum(("y", "x")) / int(mask.sum())
        series = sub.mean(("y", "x")).where(usable >= min_usable).to_pandas().dropna()
        ax.plot(series.index, series.values, marker="o", markersize=7, linewidth=2, color=colour,
                label=label)


def index_history_and_dnbr(
    nbr: xr.DataArray,
    ndvi: xr.DataArray,
    dnbr: xr.DataArray,
    burn: xr.DataArray,
    pre: tuple[str, str],
    post: tuple[str, str],
    fire: tuple[str, str],
    min_usable: float = 0.5,
) -> Figure:
    """Top and middle: mean NBR and mean NDVI per date inside the burn and over the unburned
    rest of the box, with the two windows and the fire days shaded. Bottom: the dNBR map.

    A date enters a line only if at least ``min_usable`` of that area is usable on it; the
    orbit that sees only the west of the box would otherwise produce meaningless means.
    """
    unburned = (abs(dnbr) < 0.1) & dnbr.notnull()
    masks = (
        (burn, "burned area (dNBR > 0.27)", "firebrick"),
        (unburned, "unburned area (dNBR within 0.1 of zero)", "seagreen"),
    )
    fig = plt.figure(figsize=(10, 20))
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.45])
    ax_nbr = fig.add_subplot(grid[0])
    ax_ndvi = fig.add_subplot(grid[1], sharex=ax_nbr)
    ax_map = fig.add_subplot(grid[2])
    for ax, index, name in ((ax_nbr, nbr, "NBR"), (ax_ndvi, ndvi, "NDVI")):
        _area_means(ax, index, masks, min_usable)
        top = ax.get_ylim()[1]
        for (a, b), label in ((pre, "pre-fire window"), (post, "post-fire window")):
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="grey", alpha=0.12)
            ax.text(pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2, top, label,
                    ha="center", va="top", color="#444")
        ax.axvspan(pd.Timestamp(fire[0]), pd.Timestamp(fire[1]), color="orange", alpha=0.6,
                   label="fire")
        ax.set_ylabel(f"mean {name} over the area")
        ax.grid(alpha=0.3)
    ax_nbr.set_title(f"Mean index per date, dates with at least {min_usable:.0%} of the area usable")
    ax_nbr.legend(loc="lower left")
    ax_nbr.tick_params(labelbottom=False)
    ax_ndvi.xaxis.set_major_locator(mdates.DayLocator(bymonthday=(1, 15)))
    ax_ndvi.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax_ndvi.set_xlabel("2025")
    image = ax_map.imshow(dnbr.values, **DNBR_COLOURS)
    fig.colorbar(image, ax=ax_map, orientation="horizontal", fraction=0.05, pad=0.02, aspect=40,
                 label="dNBR (red = vegetation lost)")
    ax_map.set_title("dNBR map: pre-fire median minus post-fire median")
    ax_map.set_axis_off()
    fig.tight_layout()
    return fig


def ndvi_before_after(before: xr.DataArray, after: xr.DataArray, burn: xr.DataArray) -> Figure:
    """NDVI before and after the fire, the window medians, one above the other, with the outline
    of the burn (dNBR > 0.27) on both.

    Drawn in steps of 0.1 on the colour scale of the web map: each colour is a range that can be
    read off the legend, and the picture stays a third of the size of a smooth one.
    """
    from matplotlib.colors import BoundaryNorm

    steps = np.round(np.arange(0.0, 0.81, 0.1), 1)
    cmap = plt.get_cmap(NDVI_COLOURS["cmap"], len(steps) + 1)
    norm = BoundaryNorm(steps, cmap.N, extend="both")
    fig, axes = plt.subplots(2, 1, figsize=(10, 18), layout="constrained")
    for ax, ndvi, title in ((axes[0], before, "NDVI before: median 1 July to 7 August"),
                            (axes[1], after, "NDVI after: median 13 August to 15 September")):
        image = ax.imshow(ndvi.values, cmap=cmap, norm=norm, interpolation="nearest")
        ax.contour(burn.values.astype(float), levels=[0.5], colors="firebrick", linewidths=0.8)
        ax.set_title(title)
        ax.set_axis_off()
    fig.colorbar(image, ax=axes, orientation="horizontal", fraction=0.03, pad=0.02, aspect=40, ticks=steps,
                 label="NDVI (pale = bare or burned ground, dark green = dense vegetation); red line: the burn")
    return fig


AGREEMENT_COLOURS = {"both": "#4d4d4d", "only ours": "#d7301f", "only EFFIS": "#2c7fb8"}


def agreement_map(
    ours: xr.DataArray,
    ref: xr.DataArray,
    scores: dict[str, float],
    labels: tuple[str, str, str] = tuple(AGREEMENT_COLOURS),
    title: str = "Main fire against the EFFIS burnt-area polygons, 20 m grid",
) -> Figure:
    """Two burn maps on the same grid, cropped to the fire: by default our main fire against the
    EFFIS polygons.

    Dark grey: burned in both. Red: burned only in ``ours``. Blue: burned only in ``ref``.
    ``labels`` names the three in the legend.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    a, b = ours.values.astype(bool), ref.values.astype(bool)
    codes = np.zeros(a.shape)  # 0 unburned in both
    codes[a & b] = 1
    codes[a & ~b] = 2
    codes[~a & b] = 3
    rows, cols = np.nonzero(a | b)
    margin = 25  # pixels, 500 m
    r0, r1 = max(rows.min() - margin, 0), min(rows.max() + margin, a.shape[0])
    c0, c1 = max(cols.min() - margin, 0), min(cols.max() + margin, a.shape[1])
    colours = ListedColormap(["#eef3ea", *AGREEMENT_COLOURS.values()])
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(codes[r0:r1, c0:c1], cmap=colours, vmin=0, vmax=3, interpolation="nearest")
    areas = (scores["both_ha"], scores["only_ours_ha"], scores["only_reference_ha"])
    handles = [
        Patch(color=c, label=f"{name}: {ha:,.0f} ha")
        for name, c, ha in zip(labels, AGREEMENT_COLOURS.values(), areas)
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
              title=f"IoU {scores['iou']:.2f}", frameon=False)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def burn_probability_maps(
    before: xr.DataArray,
    after: xr.DataArray,
    days: tuple[str, str],
    outlines: dict[str, tuple[xr.DataArray, str]],
) -> Figure:
    """A model's burn probability on two dates side by side, with the same outlines on both.

    ``outlines`` maps a legend label to (mask, colour). Each outline is the outer edge of its
    mask: holes are filled, so an unburned island inside a fire does not draw a line of its own.
    White: no data (cloud).
    """
    from matplotlib.lines import Line2D
    from scipy import ndimage

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for ax, day, probability in zip(axes, days, (before, after)):
        image = ax.imshow(probability, cmap="magma_r", vmin=0, vmax=1, interpolation="nearest")
        for mask, colour in outlines.values():
            ax.contour(ndimage.binary_fill_holes(mask.values), levels=[0.5], colors=colour, linewidths=1.3)
        ax.set_title(f"Burn scar probability, {day}")
        ax.set_axis_off()
    fig.colorbar(image, ax=axes, shrink=0.75, label="probability (burned above 0.5)")
    handles = [Line2D([], [], color=colour, label=label) for label, (_, colour) in outlines.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    return fig


# ESRI's legend colours, in the class order of terramind.CLASSES (0 no data ... 9 rangeland).
LAND_COVER_COLOURS = ["#ffffff", "#419bdf", "#397d49", "#7a87c6", "#e49635", "#c4281b", "#a59b8f",
                      "#a8ebff", "#616161", "#e3e2c3"]


def land_cover_maps(left: np.ndarray, right: np.ndarray, fire_10m: np.ndarray, titles: tuple[str, str],
                    names: list[str]) -> Figure:
    """Two land cover maps side by side in ESRI's colours, with the outer edge of the main fire."""
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from scipy import ndimage

    colours = ListedColormap(LAND_COVER_COLOURS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, values, title in zip(axes, (left, right), titles):
        ax.imshow(values, cmap=colours, vmin=0, vmax=len(LAND_COVER_COLOURS) - 1, interpolation="nearest")
        ax.contour(ndimage.binary_fill_holes(fire_10m), levels=[0.5], colors="black", linewidths=1.2)
        ax.set_title(title)
        ax.set_axis_off()
    present = sorted(set(np.unique(left)) | set(np.unique(right)))
    handles = [Patch(color=LAND_COVER_COLOURS[c], label=names[c]) for c in present if c != 0]
    handles.append(Line2D([], [], color="black", label="dNBR main fire"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


STEEP_COLOURS = {"low severity": "#fed976", "severe, gentler slope": "#fd8d3c", "severe and steep": "#800026"}


def slope_and_fire(
    slope: xr.DataArray,
    fire: xr.DataArray,
    severe: xr.DataArray,
    thresholds: tuple[float, ...],
    chosen: float,
) -> Figure:
    """Three plots, one above the other. Top: slope over the box with the main fire outlined.
    Middle: the main fire split into low severity, severe on gentler ground and severe and steep
    (at or above ``chosen``). Bottom: the slope distribution inside the fire by severity, with
    the candidate thresholds.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(10, 24), gridspec_kw={"height_ratios": [1.2, 1.05, 0.7]}
    )
    image = ax1.imshow(slope.values, cmap="magma_r", vmin=0, vmax=45)
    ax1.contour(fire.values.astype(float), levels=[0.5], colors="deepskyblue", linewidths=1.8)
    fig.colorbar(image, ax=ax1, orientation="horizontal", fraction=0.04, pad=0.02, aspect=40,
                 label="slope (degrees)")
    ax1.set_title("Slope on the 20 m grid; main fire in blue")
    ax1.set_axis_off()

    s = slope.values
    in_fire = fire.values.astype(bool)
    sev = severe.values.astype(bool)
    steep = s >= chosen
    ha = float(abs(slope.x[1] - slope.x[0])) ** 2 / 10_000
    codes = np.full(s.shape, np.nan)
    codes[in_fire & ~sev] = 0
    codes[in_fire & sev & ~steep] = 1
    codes[in_fire & sev & steep] = 2
    rows, cols = np.nonzero(in_fire)
    r0, r1 = max(rows.min() - 10, 0), rows.max() + 10
    c0, c1 = max(cols.min() - 10, 0), cols.max() + 10
    ax2.imshow(codes[r0:r1, c0:c1], cmap=ListedColormap(list(STEEP_COLOURS.values())),
               vmin=0, vmax=2, interpolation="nearest")
    areas = (
        (in_fire & ~sev).sum() * ha,
        (in_fire & sev & ~steep).sum() * ha,
        (in_fire & sev & steep).sum() * ha,
    )
    handles = [
        Patch(color=c, label=f"{name}: {a:,.0f} ha")
        for (name, c), a in zip(STEEP_COLOURS.items(), areas)
    ]
    ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
               title=f"steep = {chosen:.0f}° or more", frameon=False)
    ax2.set_title("Main fire: severity against slope")
    ax2.set_axis_off()

    bins = np.arange(0, 52, 2)
    bottom = np.zeros(len(bins) - 1)
    for mask, label, colour in (
        (in_fire & sev, "severe (moderate-low and above)", "#b30000"),
        (in_fire & ~sev, "low severity", "#fec44f"),
    ):
        counts, _ = np.histogram(s[mask & ~np.isnan(s)], bins=bins)
        ax3.bar(bins[:-1], counts * ha, bottom=bottom, width=2, align="edge", color=colour,
                edgecolor="white", linewidth=0.5, label=label)
        bottom += counts * ha
    for t in thresholds:
        style = "-" if t == chosen else "--"
        ax3.axvline(t, color="black", linestyle=style, linewidth=1.8 if t == chosen else 1)
        ax3.text(t, 1.01, f"{t:.0f}°", transform=ax3.get_xaxis_transform(), ha="center", va="bottom")
    ax3.set_xlabel("slope (degrees)")
    ax3.set_ylabel("hectares per 2° bin")
    ax3.set_title("Slope inside the main fire, by severity", pad=24)
    ax3.legend(loc="upper right")
    ax3.grid(alpha=0.3)
    fig.tight_layout()
    return fig


PRIORITY_EDGES = {"1: treat first": "#3b0010", "2: treat next": "#e6550d", "3: monitor": "#b0b0b0"}


def priority_map(
    cells: pd.DataFrame,
    fire: xr.DataArray,
    severe: xr.DataArray,
    slope: xr.DataArray,
    threshold: float,
    cell_m: float,
    label_top: int = 10,
) -> Figure:
    """The grid cells over the main fire, outlined by priority, the top ranks numbered, on top of
    the pixels split into low severity, severe on gentler ground and severe and steep."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle

    in_fire = fire.values.astype(bool)
    sev = severe.values.astype(bool)
    steep = slope.values >= threshold
    codes = np.full(in_fire.shape, np.nan)
    codes[in_fire & ~sev] = 0
    codes[in_fire & sev & ~steep] = 1
    codes[in_fire & sev & steep] = 2
    x, y = fire.x.values, fire.y.values
    half = abs(float(x[1] - x[0])) / 2
    extent = (x[0] - half, x[-1] + half, y[-1] - half, y[0] + half)

    fig, ax = plt.subplots(figsize=(10, 10.5))
    ax.imshow(codes, cmap=ListedColormap(list(STEEP_COLOURS.values())), vmin=0, vmax=2,
              extent=extent, interpolation="nearest", alpha=0.5)
    order = {label: i for i, label in enumerate(PRIORITY_EDGES)}
    for _, c in cells.sort_values("priority", key=lambda s: s.map(order), ascending=False).iterrows():
        first = c["priority"] == "1: treat first"
        ax.add_patch(Rectangle((c["col"] * cell_m, c["row"] * cell_m), cell_m, cell_m, fill=False,
                               edgecolor=PRIORITY_EDGES[c["priority"]],
                               linewidth=2.4 if first else 1.6 if c["priority"] == "2: treat next" else 0.6))
        if c["rank"] <= label_top:
            ax.text(c["x"], c["y"], str(c["rank"]), ha="center", va="center", fontsize=12,
                    fontweight="bold", color="white",
                    bbox={"boxstyle": "circle,pad=0.2", "facecolor": "#3b0010", "edgecolor": "none"})
    cols, rows = cells["col"], cells["row"]
    ax.set_xlim(cols.min() * cell_m - cell_m, (cols.max() + 2) * cell_m)
    ax.set_ylim(rows.min() * cell_m - cell_m, (rows.max() + 2) * cell_m)
    ax.set_aspect("equal")
    ax.set_axis_off()
    counts = cells["priority"].value_counts()
    handles = [
        Patch(facecolor="none", edgecolor=colour, linewidth=2, label=f"priority {label}: {counts.get(label, 0)} cells")
        for label, colour in PRIORITY_EDGES.items()
    ] + [Patch(color=colour, alpha=0.5, label=name) for name, colour in STEEP_COLOURS.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    ax.set_title(f"{cell_m:.0f} m cells ranked by severe and steep share (steep = {threshold:.0f}° or more);"
                 f" top {label_top} numbered")
    fig.tight_layout()
    return fig


WEB_CRS = "EPSG:3857"  # Web Mercator, the projection of Leaflet and most web maps


class WebLayer(NamedTuple):
    """An image ready for the web map."""

    name: str
    url: str  # the image itself as a data URI, so the saved notebook carries it
    corners: list[list[float]]  # [[south, west], [north, east]], what Leaflet asks for
    shown: bool = False  # switched on when the map opens


def to_web(values: np.ndarray, like: xr.DataArray) -> tuple[np.ndarray, list[list[float]]]:
    """Move an image from the cube grid onto the grid of a web map, and give its corners.

    Leaflet places an image by stretching it between two corners in latitude and longitude on a
    Web Mercator map. The cube is in UTM 33N, turned about 0.4° against north here, so stretched
    as it is the image would sit up to 80 m (4 pixels) off at the edges. Reprojected first, every
    pixel lands where it belongs. The new grid keeps 20 m on the ground, so each new pixel takes
    the nearest old one: 99.6 % of the pixels appear exactly once, and every colour on the map is
    a value that was measured, never an average. ``values`` is (y, x) or (y, x, bands) on the
    grid of ``like``; the result is NaN outside the cube.
    """
    data = np.asarray(values, dtype="float32")
    data = data[np.newaxis] if data.ndim == 2 else np.moveaxis(data, -1, 0)
    grid = xr.DataArray(data, dims=("band", "y", "x"), coords={"y": like.y.values, "x": like.x.values})
    grid = grid.rio.write_crs(like.rio.crs).rio.write_nodata(np.nan)
    web = grid.rio.reproject(WEB_CRS, resampling=Resampling.nearest)
    west, south, east, north = transform_bounds(WEB_CRS, "EPSG:4326", *web.rio.bounds())
    out = np.moveaxis(web.values, 0, -1)
    return (out[..., 0] if out.shape[-1] == 1 else out), [[south, west], [north, east]]


def data_uri(pixels: np.ndarray, fmt: str) -> str:
    """An 8-bit image as a data URI. WebP for smooth images (the pictures, dNBR): compressed with
    small losses, like JPEG, to about a fifth of a PNG, and unlike JPEG it keeps transparency.
    PNG for the class layers: lossless, so every class keeps its exact colour."""
    buffer = io.BytesIO()
    options = {"quality": 85} if fmt == "WEBP" else {"optimize": True}
    Image.fromarray(pixels).save(buffer, fmt, **options)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buffer.getvalue()).decode()}"


def _rgba(colour: str) -> tuple[int, ...]:
    return tuple(round(255 * v) for v in to_rgba(colour))


def picture_layers(refl: xr.Dataset, pre_day: str, post_day: str) -> list[WebLayer]:
    """The four pictures of step 4 for the web map: true colour and short-wave infrared colour,
    before and after, with the same stretch (from the pre-fire day)."""
    pre = refl.sel(time=pre_day).squeeze("time")
    post = refl.sel(time=post_day).squeeze("time")
    layers = []
    for label, bands in (("true colour", TRUE_COLOUR), ("short-wave infrared colour", SWIR_COLOUR)):
        bounds = stretch_bounds(pre, bands)
        for day, ds_day in ((pre_day, pre), (post_day, post)):
            rgb, corners = to_web(to_rgb(ds_day, bands, bounds), ds_day[bands[0]])
            pixels = (255 * np.where(np.isnan(rgb), MISSING_GREY, rgb)).round().astype("uint8")
            layers.append(WebLayer(f"{label}, {day}", data_uri(pixels, "WEBP"), corners))
    return layers


def _smooth_layer(name: str, da: xr.DataArray, colours: dict) -> WebLayer:
    """A continuous quantity (an index) painted on a colour scale, transparent where it is NaN."""
    values, corners = to_web(da.values, da)
    norm = Normalize(colours["vmin"], colours["vmax"])
    rgba = colormaps[colours["cmap"]](norm(values), bytes=True)
    rgba[..., 3] = np.where(np.isnan(values), 0, 255)
    return WebLayer(name, data_uri(rgba, "WEBP"), corners)


def index_layers(ndvi_before: xr.DataArray, ndvi_after: xr.DataArray, dnbr: xr.DataArray) -> list[WebLayer]:
    """NDVI before and after (window medians) and dNBR, in the colours of step 5. They cover the
    whole box, so the map shows them as backgrounds: one at a time and opaque, so a colour always
    means the same value, whatever lies underneath."""
    return [
        _smooth_layer("NDVI before (1 Jul to 7 Aug median)", ndvi_before, NDVI_COLOURS),
        _smooth_layer("NDVI after (13 Aug to 15 Sep median)", ndvi_after, NDVI_COLOURS),
        _smooth_layer("dNBR", dnbr, DNBR_COLOURS),
    ]


def class_layers(severity: xr.DataArray, severe_steep: xr.DataArray) -> list[WebLayer]:
    """The severity classes of the main fire and the severe-and-steep ground, in the colours of
    steps 6 and 8. Transparent where there is nothing to show (unburned ground, outside the box),
    so they sit on top of any background."""
    codes, corners = to_web(np.where(severity.values == 255, np.nan, severity.values), severity)
    classes = np.zeros(codes.shape + (4,), dtype="uint8")
    for code, colour in enumerate(SEVERITY_COLOURS[1:], start=1):  # 0, unburned, stays transparent
        classes[codes == code] = _rgba(colour)

    hit, _ = to_web(severe_steep.values, severity)
    steep = np.zeros(hit.shape + (4,), dtype="uint8")
    steep[hit == 1] = _rgba(STEEP_COLOURS["severe and steep"])
    return [
        WebLayer("severity classes, main fire", data_uri(classes, "PNG"), corners, shown=True),
        WebLayer("severe and steep ground", data_uri(steep, "PNG"), corners),
    ]


def interactive_map(perimeter_file, cells_file, backgrounds=(), layers=(), effis=()):
    """A small web map drawn from the written files and the layers of the notebook.

    Backgrounds, one at a time: satellite, streets, and the images in ``backgrounds`` (the
    pictures of step 4, NDVI, dNBR), so one click goes from before to after. On top, each switched
    on and off: the class ``layers``, the ``effis`` burnt-area polygons as dashed outlines, the
    flagged cells coloured by priority with their numbers on hover, and the main fire outline.

    Interactive in Jupyter, VS Code and Colab; GitHub does not run it, so the notebook also keeps
    the static priority map.
    """
    import folium
    import geopandas as gpd

    perimeter = gpd.read_file(perimeter_file)
    cells = gpd.read_file(cells_file)
    west, south, east, north = perimeter.total_bounds
    fmap = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=14, tiles=None,
                      control_scale=True)
    folium.TileLayer("Esri.WorldImagery", name="satellite (Esri)").add_to(fmap)
    folium.TileLayer("OpenStreetMap", name="streets (OpenStreetMap)").add_to(fmap)
    for layer in backgrounds:  # overlay=False: a background, chosen with the radio buttons
        folium.raster_layers.ImageOverlay(layer.url, layer.corners, name=layer.name, overlay=False,
                                          show=False).add_to(fmap)
    for layer in layers:
        folium.raster_layers.ImageOverlay(layer.url, layer.corners, name=layer.name, opacity=0.8,
                                          show=layer.shown).add_to(fmap)
    if effis:
        folium.GeoJson(
            {"type": "FeatureCollection", "features": list(effis)},
            name="EFFIS burnt areas",
            show=False,
            style_function=lambda f: {"color": AGREEMENT_COLOURS["only EFFIS"], "weight": 2,
                                      "dashArray": "6 4", "fill": False},
            tooltip=folium.GeoJsonTooltip(fields=["COMMUNE", "FIREDATE", "AREA_HA"],
                                          aliases=["municipality", "start", "hectares"]),
        ).add_to(fmap)
    fill = {"1: treat first": "#67000d", "2: treat next": "#fd8d3c"}
    folium.GeoJson(
        cells,
        name="flagged cells",
        style_function=lambda f: {"color": fill[f["properties"]["priority"]], "weight": 2,
                                  "fillColor": fill[f["properties"]["priority"]], "fillOpacity": 0.4},
        tooltip=folium.GeoJsonTooltip(
            fields=["rank", "priority", "severe_steep_share", "severe_steep_ha", "green_ndvi", "lat", "lon"],
            aliases=["rank", "priority", "share severe and steep", "severe and steep (ha)",
                     "green cover (NDVI)", "latitude", "longitude"],
        ),
    ).add_to(fmap)
    folium.GeoJson(
        perimeter,
        name="main fire perimeter",
        style_function=lambda f: {"color": "#00b4ff", "weight": 2, "fill": False},
    ).add_to(fmap)
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.fit_bounds([[south, west], [north, east]])
    return fmap
