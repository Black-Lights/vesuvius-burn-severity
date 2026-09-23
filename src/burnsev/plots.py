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


AGREEMENT_COLOURS = {"both": "#4d4d4d", "only ours": "#d7301f", "only EFFIS": "#2c7fb8"}


def agreement_map(ours: xr.DataArray, ref: xr.DataArray, scores: dict[str, float]) -> Figure:
    """Our main fire against the EFFIS polygons on the same grid, cropped to the fire.

    Dark grey: burned in both. Red: burned only in ours. Blue: burned only in EFFIS.
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
        for (name, c), ha in zip(AGREEMENT_COLOURS.items(), areas)
    ]
    ax.legend(handles=handles, loc="lower left", title=f"IoU {scores['iou']:.2f}")
    ax.set_title("Main fire against the EFFIS burnt-area polygons, 20 m grid")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


STEEP_COLOURS = {"low severity": "#fed976", "severe, gentler slope": "#fd8d3c", "severe and steep": "#800026"}


def slope_and_fire(
    slope: xr.DataArray,
    fire: xr.DataArray,
    severe: xr.DataArray,
    thresholds: tuple[float, ...],
    chosen: float,
) -> Figure:
    """Left: slope over the box with the main fire outlined. Middle: the main fire split into
    low severity, severe on gentler ground and severe and steep (at or above ``chosen``).
    Right: the slope distribution inside the fire by severity, with the candidate thresholds.
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(21, 6.5), gridspec_kw={"width_ratios": [1.15, 1, 1]}
    )
    image = ax1.imshow(slope.values, cmap="magma_r", vmin=0, vmax=45)
    ax1.contour(fire.values.astype(float), levels=[0.5], colors="deepskyblue", linewidths=1.8)
    fig.colorbar(image, ax=ax1, shrink=0.8, label="slope (degrees)")
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
    ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1,
               title=f"steep = {chosen:.0f}° or more")
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
