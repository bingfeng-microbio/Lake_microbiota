# -*- coding: utf-8 -*-

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.patches import Patch


# =============================================================================
# 1. FILE PATHS
# =============================================================================

BASE_DIR = Path("/Users/bingfengchen/Desktop/Freshwater")

INPUT_FILE = (
    BASE_DIR
    / "PC_convex_hull_extrapolation/"
      "extrapolation_layer_for_bivariate_and_extended_data.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "C_convex_hull_extrapolation/Robinson_maps"
)


# =============================================================================
# 2. FIGURE AND MAP SETTINGS
# =============================================================================

FIGSIZE = (14, 9)
PNG_DPI = 1200
SAVE_PDF = True
SHOW_TITLE = True
DRAW_GRIDLINES = True
EXCLUDE_ANTARCTICA = True

MAP_MIN_LATITUDE = -60.0
MAP_MAX_LATITUDE = 90.0

POINT_SIZE = 1.0
POINT_ALPHA = 1.0
POINT_MARKER = "s"

DISTANCE_DISPLAY_UPPER_QUANTILE = 0.99

LAND_COLOR = "#F5F5F5"
OCEAN_COLOR = "white"
COAST_COLOR = "black"

COASTLINE_WIDTH = 0.50
COASTLINE_ZORDER = 5

GRID_LONGITUDES = [-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]
GRID_LATITUDES = [-30, 30, 60]

GRIDLINE_COLOR = "black"
GRIDLINE_WIDTH = 0.50
GRIDLINE_ALPHA = 0.2
GRIDLINE_STYLE = (0, (2, 2))

SIDE_LINE_WIDTH = GRIDLINE_WIDTH
SIDE_LINE_ALPHA = GRIDLINE_ALPHA
SIDE_LINE_STYLE = GRIDLINE_STYLE

EQUATOR_LINE_WIDTH = GRIDLINE_WIDTH
EQUATOR_LINE_ALPHA = GRIDLINE_ALPHA
EQUATOR_LINE_STYLE = GRIDLINE_STYLE

RETENTION_LEGEND_LOC = "center left"
RETENTION_LEGEND_ANCHOR = (1.02, 0.50)
RETENTION_LEGEND_FONT_SIZE = 8.5


# =============================================================================
# 3. INPUT COLUMNS
# =============================================================================

LON_COL = "Longitude"
LAT_COL = "Latitude"
ENV_COL = "PCA_extrapolation_mean"
DISTANCE_COL = "Nearest_training_location_km"
COMBINED_COL = "Combined_extrapolation_index_selected"
COMBINED_CLASS_COL = "Combined_extrapolation_class_5"
RETAIN_COL = "Retain_support_ge_50pct_expected_months"

REQUIRED_COLUMNS = [
    LON_COL,
    LAT_COL,
    ENV_COL,
    DISTANCE_COL,
    COMBINED_COL,
    COMBINED_CLASS_COL,
    RETAIN_COL,
]


# =============================================================================
# 4. COLOUR SETTINGS
# =============================================================================

plt.style.use("ggplot")

mpl.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

EXTRAPOLATION_CMAP = LinearSegmentedColormap.from_list(
    "lake_extrapolation",
    [
        "#FFFBE6",
        "#FDE0A3",
        "#F6A06A",
        "#E45B75",
        "#9B3C8E",
        "#4B0C6B",
    ],
    N=256,
)

REVERSED_EXTRAPOLATION_CMAP = EXTRAPOLATION_CMAP.reversed(
    name="lake_extrapolation_low_dark_high_pale"
)

QUINTILE_COLORS = [
    "#FFFBE6",
    "#F6C77A",
    "#EC7B70",
    "#B5448C",
    "#4B0C6B",
]
QUINTILE_CMAP = ListedColormap(QUINTILE_COLORS)

RETENTION_COLORS = {
    "Excluded (<50% supported months)": "#FFF100",
    "Retained (≥50% supported months)": "#571267",
}


# =============================================================================
# 5. DATA
# =============================================================================

def require_columns(data, columns):
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise KeyError(
            "Input file is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing)
        )


def normalize_longitude(values):
    numeric = pd.to_numeric(values, errors="coerce")
    return ((numeric + 180.0) % 360.0) - 180.0


def parse_boolean(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()

    result = pd.Series(False, index=series.index, dtype=bool)
    result.loc[numeric == 1] = True
    result.loc[text.isin({"true", "t", "yes", "y"})] = True
    return result


def load_plot_data():
    print(f"Reading: {INPUT_FILE}")
    data = pd.read_csv(INPUT_FILE, low_memory=False)
    require_columns(data, REQUIRED_COLUMNS)

    data[LON_COL] = normalize_longitude(data[LON_COL])
    data[LAT_COL] = pd.to_numeric(data[LAT_COL], errors="coerce")

    for column in [ENV_COL, DISTANCE_COL, COMBINED_COL, COMBINED_CLASS_COL]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data[RETAIN_COL] = parse_boolean(data[RETAIN_COL])

    valid_coordinates = (
        data[LON_COL].between(-180, 180, inclusive="both")
        & data[LAT_COL].between(-90, 90, inclusive="both")
    )
    data = data.loc[valid_coordinates].copy()

    if EXCLUDE_ANTARCTICA:
        data = data.loc[data[LAT_COL] >= MAP_MIN_LATITUDE].copy()

    print(f"Valid global lakes: {len(data):,}")
    print(
        f"Retained lakes: {data[RETAIN_COL].sum():,}/"
        f"{len(data):,} ({data[RETAIN_COL].mean() * 100:.2f}%)"
    )

    return data


# =============================================================================
# 6. MAP CONSTRUCTION
# =============================================================================

def hide_default_map_outline(ax):
    try:
        ax.spines["geo"].set_visible(False)
    except Exception:
        pass

    try:
        ax.outline_patch.set_visible(False)
    except Exception:
        pass

    try:
        ax.patch.set_edgecolor("none")
        ax.patch.set_linewidth(0.0)
    except Exception:
        pass


def draw_projected_line(
    ax,
    projection,
    longitudes,
    latitudes,
    *,
    linewidth,
    alpha,
    linestyle,
    zorder,
):
    points = projection.transform_points(
        ccrs.PlateCarree(),
        longitudes.astype(float),
        latitudes.astype(float),
    )

    x = points[:, 0]
    y = points[:, 1]
    valid = np.isfinite(x) & np.isfinite(y)

    ax.plot(
        x[valid],
        y[valid],
        transform=projection,
        color=GRIDLINE_COLOR,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        zorder=zorder,
        clip_on=False,
        solid_capstyle="butt",
    )


def add_graticules(ax, projection):
    if not DRAW_GRIDLINES:
        return

    gridliner = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=GRIDLINE_WIDTH,
        color=GRIDLINE_COLOR,
        alpha=GRIDLINE_ALPHA,
        linestyle=GRIDLINE_STYLE,
        x_inline=False,
        y_inline=False,
        zorder=2,
    )
    gridliner.xlocator = mticker.FixedLocator(GRID_LONGITUDES)
    gridliner.ylocator = mticker.FixedLocator(GRID_LATITUDES)

    side_latitudes = np.linspace(
        MAP_MIN_LATITUDE if EXCLUDE_ANTARCTICA else -89.999,
        MAP_MAX_LATITUDE if EXCLUDE_ANTARCTICA else 89.999,
        2001,
    )

    for side_longitude in (-179.999999, 179.999999):
        draw_projected_line(
            ax,
            projection,
            np.full_like(side_latitudes, side_longitude),
            side_latitudes,
            linewidth=SIDE_LINE_WIDTH,
            alpha=SIDE_LINE_ALPHA,
            linestyle=SIDE_LINE_STYLE,
            zorder=100,
        )

    equator_longitudes = np.linspace(-179.999999, 179.999999, 4001)
    draw_projected_line(
        ax,
        projection,
        equator_longitudes,
        np.zeros_like(equator_longitudes),
        linewidth=EQUATOR_LINE_WIDTH,
        alpha=EQUATOR_LINE_ALPHA,
        linestyle=EQUATOR_LINE_STYLE,
        zorder=99,
    )


def create_robinson_axes():
    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    projection = ccrs.Robinson(central_longitude=0)
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    if EXCLUDE_ANTARCTICA:
        ax.set_extent(
            [-180.0, 180.0, MAP_MIN_LATITUDE, MAP_MAX_LATITUDE],
            crs=ccrs.PlateCarree(),
        )
    else:
        ax.set_global()

    ax.add_feature(
        cfeature.OCEAN,
        facecolor=OCEAN_COLOR,
        edgecolor="none",
        zorder=0,
    )
    ax.add_feature(
        cfeature.LAND,
        facecolor=LAND_COLOR,
        edgecolor="none",
        zorder=1,
    )

    hide_default_map_outline(ax)
    add_graticules(ax, projection)

    ax.add_feature(
        cfeature.COASTLINE,
        edgecolor=COAST_COLOR,
        linewidth=COASTLINE_WIDTH,
        zorder=COASTLINE_ZORDER,
    )

    return fig, ax


# =============================================================================
# 7. FIGURE HELPERS
# =============================================================================

def add_colorbar(
    fig,
    ax,
    mappable,
    label,
    ticks=None,
    ticklabels=None,
    extend="neither",
):
    colorbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation="vertical",
        pad=0.05,
        aspect=30,
        extend=extend,
    )
    colorbar.set_label(label, fontsize=10)
    colorbar.ax.tick_params(labelsize=8)

    if ticks is not None:
        colorbar.set_ticks(ticks)
    if ticklabels is not None:
        colorbar.ax.set_yticklabels(ticklabels)


def save_figure(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    png_path = OUTPUT_DIR / f"{stem}.png"
    fig.savefig(
        png_path,
        format="png",
        dpi=PNG_DPI,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
    )
    print(f"Saved: {png_path}")

    if SAVE_PDF:
        pdf_path = OUTPUT_DIR / f"{stem}.pdf"
        fig.savefig(
            pdf_path,
            format="pdf",
            dpi=PNG_DPI,
            bbox_inches="tight",
            pad_inches=0.05,
            facecolor="white",
        )
        print(f"Saved: {pdf_path}")

    plt.close(fig)


# =============================================================================
# 8. MAP FUNCTIONS
# =============================================================================

def draw_continuous_map(
    data,
    value_col,
    stem,
    colorbar_label,
    vmin,
    vmax,
    title,
    *,
    cmap=EXTRAPOLATION_CMAP,
    extend="neither",
    sort_ascending=True,
):
    plot_data = data[[LON_COL, LAT_COL, value_col]].dropna().copy()
    if plot_data.empty:
        raise ValueError(f"No valid values found for {value_col}.")

    plot_data = plot_data.sort_values(value_col, ascending=sort_ascending)

    fig, ax = create_robinson_axes()
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    scatter = ax.scatter(
        plot_data[LON_COL].to_numpy(dtype=float),
        plot_data[LAT_COL].to_numpy(dtype=float),
        c=plot_data[value_col].to_numpy(dtype=float),
        cmap=cmap,
        norm=norm,
        alpha=POINT_ALPHA,
        s=POINT_SIZE,
        linewidths=0,
        edgecolors="none",
        marker=POINT_MARKER,
        transform=ccrs.PlateCarree(),
        zorder=3,
        rasterized=True,
    )

    add_colorbar(fig, ax, scatter, colorbar_label, extend=extend)

    if SHOW_TITLE:
        ax.set_title(title, fontsize=16, weight="bold")

    save_figure(fig, stem)


def draw_quintile_map(data):
    plot_data = data[[LON_COL, LAT_COL, COMBINED_CLASS_COL]].dropna().copy()
    plot_data[COMBINED_CLASS_COL] = plot_data[COMBINED_CLASS_COL].astype(int)
    plot_data = plot_data.loc[
        plot_data[COMBINED_CLASS_COL].between(1, 5, inclusive="both")
    ].sort_values(COMBINED_CLASS_COL)

    fig, ax = create_robinson_axes()
    boundaries = np.arange(0.5, 6.0, 1.0)
    norm = BoundaryNorm(boundaries, QUINTILE_CMAP.N)

    scatter = ax.scatter(
        plot_data[LON_COL].to_numpy(dtype=float),
        plot_data[LAT_COL].to_numpy(dtype=float),
        c=plot_data[COMBINED_CLASS_COL].to_numpy(dtype=int),
        cmap=QUINTILE_CMAP,
        norm=norm,
        alpha=POINT_ALPHA,
        s=POINT_SIZE,
        linewidths=0,
        edgecolors="none",
        marker=POINT_MARKER,
        transform=ccrs.PlateCarree(),
        zorder=3,
        rasterized=True,
    )

    add_colorbar(
        fig,
        ax,
        scatter,
        "Combined extrapolation quintile",
        ticks=[1, 2, 3, 4, 5],
        ticklabels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"],
    )

    if SHOW_TITLE:
        ax.set_title("Combined extrapolation classes", fontsize=16, weight="bold")

    save_figure(fig, "04_combined_extrapolation_quintiles_all_lakes_Robinson")


def draw_retention_map(data):
    excluded = data.loc[~data[RETAIN_COL], [LON_COL, LAT_COL]].dropna()
    retained = data.loc[data[RETAIN_COL], [LON_COL, LAT_COL]].dropna()

    fig, ax = create_robinson_axes()

    ax.scatter(
        excluded[LON_COL].to_numpy(dtype=float),
        excluded[LAT_COL].to_numpy(dtype=float),
        c=RETENTION_COLORS["Excluded (<50% supported months)"],
        alpha=POINT_ALPHA,
        s=POINT_SIZE,
        linewidths=0,
        edgecolors="none",
        marker=POINT_MARKER,
        transform=ccrs.PlateCarree(),
        zorder=3,
        rasterized=True,
    )

    ax.scatter(
        retained[LON_COL].to_numpy(dtype=float),
        retained[LAT_COL].to_numpy(dtype=float),
        c=RETENTION_COLORS["Retained (≥50% supported months)"],
        alpha=POINT_ALPHA,
        s=POINT_SIZE,
        linewidths=0,
        edgecolors="none",
        marker=POINT_MARKER,
        transform=ccrs.PlateCarree(),
        zorder=4,
        rasterized=True,
    )

    legend_handles = [
        Patch(
            facecolor=RETENTION_COLORS["Retained (≥50% supported months)"],
            edgecolor="none",
            label=f"Retained: {len(retained):,}",
        ),
        Patch(
            facecolor=RETENTION_COLORS["Excluded (<50% supported months)"],
            edgecolor="none",
            label=f"Excluded: {len(excluded):,}",
        ),
    ]

    legend = ax.legend(
        handles=legend_handles,
        loc=RETENTION_LEGEND_LOC,
        bbox_to_anchor=RETENTION_LEGEND_ANCHOR,
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#666666",
        fontsize=RETENTION_LEGEND_FONT_SIZE,
        handlelength=1.2,
        borderpad=0.6,
        borderaxespad=0.0,
    )
    legend.get_frame().set_linewidth(0.5)

    fig.subplots_adjust(right=0.82)

    if SHOW_TITLE:
        ax.set_title(
            "Environmental-domain support for main analyses",
            fontsize=16,
            weight="bold",
        )

    save_figure(fig, "05_environmental_support_retained_excluded_Robinson")


# =============================================================================
# 9. MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_plot_data()

    environment_max = float(data[ENV_COL].dropna().max())
    environment_vmax = float(np.ceil(environment_max * 10.0) / 10.0)

    draw_continuous_map(
        data=data,
        value_col=ENV_COL,
        stem="01_PCA_environmental_extrapolation_all_lakes_Robinson",
        colorbar_label="Environmental extrapolation",
        vmin=0.0,
        vmax=environment_vmax,
        title="PCA convex-hull environmental extrapolation",
        cmap=REVERSED_EXTRAPOLATION_CMAP,
        sort_ascending=False,
    )

    distance_vmax = float(
        data[DISTANCE_COL].dropna().quantile(DISTANCE_DISPLAY_UPPER_QUANTILE)
    )

    draw_continuous_map(
        data=data,
        value_col=DISTANCE_COL,
        stem="02_nearest_training_lake_distance_all_lakes_Robinson",
        colorbar_label="Distance to nearest training lake (km)",
        vmin=0.0,
        vmax=distance_vmax,
        title="Geographical sampling distance",
        extend="max",
    )

    combined_max = float(data[COMBINED_COL].dropna().max())
    combined_vmax = float(np.ceil(combined_max * 10.0) / 10.0)

    draw_continuous_map(
        data=data,
        value_col=COMBINED_COL,
        stem="03_combined_extrapolation_Distance2_Environment1_all_lakes_Robinson",
        colorbar_label="Combined extrapolation index",
        vmin=0.0,
        vmax=combined_vmax,
        title="Combined extrapolation (distance:environment = 2:1)",
        cmap=REVERSED_EXTRAPOLATION_CMAP,
        sort_ascending=False,
    )

    draw_quintile_map(data)
    draw_retention_map(data)

    retained = data.loc[data[RETAIN_COL]].copy()

    draw_continuous_map(
        data=retained,
        value_col=COMBINED_COL,
        stem="06_combined_extrapolation_retained_main_analysis_lakes_Robinson",
        colorbar_label="Combined extrapolation index",
        vmin=0.0,
        vmax=combined_vmax,
        title="Combined extrapolation for retained lakes",
        cmap=REVERSED_EXTRAPOLATION_CMAP,
        sort_ascending=False,
    )

    print("\nFinished all Robinson maps.")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
