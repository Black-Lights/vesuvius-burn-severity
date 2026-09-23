"""Pictures of the cube. Display only: nothing here changes a number in the analysis."""

from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "legend.fontsize": 12})

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
    fig, ax = plt.subplots(figsize=(12, 3.8))
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
    ax.legend(loc="lower right", ncol=4)
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for (label, bands), row in zip(rows, axes):
        bounds = stretch_bounds(pre, bands)
        for ax, (day, ds_day) in zip(row, ((pre_day, pre), (post_day, post))):
            ax.imshow(to_rgb(ds_day, bands, bounds))
            usable = float(ds_day["valid_fraction"]) * 100
            ax.set_title(f"{label}, {day} ({usable:.0f}% usable)")
            ax.set_axis_off()
    fig.tight_layout()
    return fig


SEVERITY_COLOURS = ["#eef3ea", "#ffffb2", "#fd8d3c", "#e31a1c", "#67000d"]  # unburned to high
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
    ax.legend(handles=handles, loc="lower left", title="dNBR class, Key and Benson (2006)")
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
    """Left: mean NBR (top) and mean NDVI (bottom) per date inside the burn and over the
    unburned rest of the box, with the two windows and the fire days shaded. Right: the dNBR map.

    A date enters a line only if at least ``min_usable`` of that area is usable on it; the
    orbit that sees only the west of the box would otherwise produce meaningless means.
    """
    unburned = (abs(dnbr) < 0.1) & dnbr.notnull()
    masks = (
        (burn, "burned area (dNBR > 0.27)", "firebrick"),
        (unburned, "unburned area (dNBR within 0.1 of zero)", "seagreen"),
    )
    fig = plt.figure(figsize=(17, 10))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
    ax_nbr = fig.add_subplot(grid[0, 0])
    ax_ndvi = fig.add_subplot(grid[1, 0], sharex=ax_nbr)
    ax_map = fig.add_subplot(grid[:, 1])
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
    ax_nbr.set_title(f"Mean index per date (dates with at least {min_usable:.0%} of the area usable)")
    ax_nbr.legend(loc="lower left")
    ax_nbr.tick_params(labelbottom=False)
    ax_ndvi.xaxis.set_major_locator(mdates.DayLocator(bymonthday=(1, 15)))
    ax_ndvi.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax_ndvi.set_xlabel("2025")
    image = ax_map.imshow(dnbr.values, cmap="RdYlGn_r", vmin=-0.3, vmax=0.9)
    fig.colorbar(image, ax=ax_map, shrink=0.6, label="dNBR (red = vegetation lost)")
    ax_map.set_title("dNBR map: pre-fire median minus post-fire median")
    ax_map.set_axis_off()
    fig.tight_layout()
    return fig
