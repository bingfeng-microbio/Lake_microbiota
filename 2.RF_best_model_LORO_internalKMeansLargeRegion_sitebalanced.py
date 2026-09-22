# -*- coding: utf-8 -*-
"""
Random Forest Leave-One-Region-Out cross-validation based on previous model-comparison output

Purpose
-------
This script reads the output table from your previous model-comparison script,
selects the Random Forest best hyperparameters for each target variable, and then
performs Leave-One-Region-Out cross-validation (LORO-CV).

Recommended for the current lake microbiome dataset
---------------------------------------------------
Your dataset contains:
- Latitude
- Longitude
- Continent
- climate_zone, e.g. Dfb, Cfa, Aw

Rare geo-climatic regions with fewer than MIN_REGION_UNIQUE_SITES unique sampling
sites are merged into "Other_small_regions". Very large geo-climatic regions with
more than MAX_REGION_UNIQUE_SITES unique sites are first split into KMeans spatial
subregions. This prevents a single oversized region from dominating LORO-CV.

Key logic
---------
1. Read original lake microbiome data.
2. Read previous output Excel, sheet "Model Results".
3. Keep rows where Model == "Random Forest".
4. For each target variable, parse saved Random Forest best parameters.
5. Construct geo-climatic regions from Continent and climate_zone.
6. Split oversized regions into KMeans spatial subregions based on unique-site counts.
7. Merge poorly represented regions based on unique-site counts.
8. For each final region, train the RF model on all other regions and test on the held-out region.
8. Calculate sample-level and site-balanced OOF R2, RMSE, MAE.
9. Save LORO summary, fold metrics, OOF predictions, region diagnostics, feature importance, and final RF models.

"""

import os
import re
import ast
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# 1. Configuration
# =============================================================================

# Original data used for model fitting.
DATA_PATH = "data_for_ML.xlsx"
DATA_SHEET_NAME = 0

# Previous output from your model-comparison script.
# It must contain a sheet named "Model Results" with columns:
# Target Variable, Model, Best Parameters, R2_test, etc.
PREVIOUS_RESULTS_EXCEL = "hyperparameter_tuning.xlsx"
PREVIOUS_RESULTS_SHEET = "Model Results"

OUTPUT_DIR = "RF_LORO_geoclimatic_KMeansLargeRegion_sitebalanced_from_best_results"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "final_RF_models")

RANDOM_STATE = 43
N_JOBS = -1

# Region settings.
CONTINENT_COL = "Continent"
CLIMATE_COL = "climate_zone"
REGION_COL_OUT = "GeoClimate_Region"
SMALL_REGION_LABEL = "Other_small_regions"

# Merge regions with fewer than this many unique coordinate sites.
# Based on your current dataset, 20 unique sites keeps the main regions and merges small unstable regions.
MIN_REGION_UNIQUE_SITES = 20

# Split very large geo-climatic regions before LORO-CV.
# This prevents one oversized region, e.g. North_America_D, from dominating the whole validation.
# The split is based on unique sampling sites, not sample counts.
SPLIT_LARGE_REGIONS = True
MAX_REGION_UNIQUE_SITES = 150
LARGE_REGION_SPLIT_METHOD = "internal_kmeans_spherical_xyz"

# Fold diagnostics.
MIN_TEST_SITES_WARN = 10
MIN_TEST_SAMPLES_WARN = 30

# Site definition.
LAT_COL = "Latitude"
LON_COL = "Longitude"
SITE_COORD_DECIMALS = 6

# Predictor columns used in your lake microbiome model.
# Latitude / Longitude / Continent / climate_zone are intentionally excluded from predictors.
FACTOR_COLS = [
    "si10", "sp", "ssr", "t2m", "tp",
    "Lake_area", "Shore_dev", "Dis_avg", "Res_time", "Elevation",
    "Cropland", "Forest", "Grassland","Urban", "Barren"
]

# If None, target variables will be inferred from the previous Model Results sheet
# after filtering Model == "Random Forest".
# If you want to force specific targets, replace None with a list, e.g.:
# TARGET_COLS = ["Shannon_arc", "Shannon_bac", ..., "Sulfur_oxidation"]
TARGET_COLS = None

ID_COL_CANDIDATES = ["SampleID"]


# =============================================================================
# 2. Utility functions
# =============================================================================

def safe_filename(text: str) -> str:
    """Make a string safe for filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calc_metrics(y_true, y_pred) -> dict:
    """Return R2, RMSE, and MAE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) < 2:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan}

    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def check_required_columns(df: pd.DataFrame, required_cols, object_name: str):
    """Raise an informative error if required columns are missing."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{object_name} is missing required columns:\n"
            + "\n".join(missing)
        )


def find_id_col(df: pd.DataFrame):
    """Find a likely sample ID column if present."""
    for c in ID_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


def normalize_longitude(lon_values):
    """Normalize longitudes to [-180, 180)."""
    lon_values = np.asarray(lon_values, dtype=float)
    return ((lon_values + 180) % 360) - 180


def make_site_id(data_t: pd.DataFrame) -> pd.Series:
    """Create unique Site_ID based on rounded latitude and normalized longitude."""
    lat = pd.to_numeric(data_t[LAT_COL], errors="coerce").round(SITE_COORD_DECIMALS)
    lon = pd.Series(
        normalize_longitude(pd.to_numeric(data_t[LON_COL], errors="coerce")),
        index=data_t.index,
    ).round(SITE_COORD_DECIMALS)
    return lat.astype(str) + "_" + lon.astype(str)


def parse_random_forest_params(best_params_string) -> dict:
    """
    Parse Random Forest best parameters saved by GridSearchCV.

    Expected examples:
    "{'randomforest__max_depth': 20, 'randomforest__min_samples_split': 2, 'randomforest__n_estimators': 300}"
    "{'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 300}"

    Returned example:
    {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 300}
    """
    if pd.isna(best_params_string):
        return {}

    if isinstance(best_params_string, dict):
        raw = best_params_string
    else:
        raw = ast.literal_eval(str(best_params_string))

    rf_params = {}
    for key, value in raw.items():
        if key.startswith("randomforest__"):
            new_key = key.replace("randomforest__", "")
            rf_params[new_key] = value
        elif "__" not in key:
            # Allows already-clean parameter names if needed.
            # Ignore non-RF metadata such as model_type.
            if key not in {"model_type", "model", "algorithm"}:
                rf_params[key] = value

    # Avoid conflict with fixed n_jobs/random_state below.
    rf_params.pop("n_jobs", None)
    rf_params.pop("random_state", None)

    return rf_params


def make_rf_pipeline(rf_params: dict) -> Pipeline:
    """
    Build Random Forest pipeline consistent with the previous model-comparison script.
    Scaling is retained for consistency, although RF itself does not require scaling.
    """
    rf = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        **rf_params
    )

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("randomforest", rf),
    ])


def clean_text_series(s: pd.Series, missing_label="Unknown") -> pd.Series:
    """Convert a text column to stripped strings and replace empty/nan-like values."""
    out = s.astype(str).str.strip()
    out = out.replace({"": missing_label, "nan": missing_label, "NaN": missing_label, "None": missing_label})
    return out


def make_geoclimatic_region(data_t: pd.DataFrame) -> pd.Series:
    """
    Construct region = cleaned Continent + '_' + major Köppen climate class.
    Example: North America + Dfb -> North America_D.
    """
    continent = clean_text_series(data_t[CONTINENT_COL], missing_label="Unknown_continent")
    climate = clean_text_series(data_t[CLIMATE_COL], missing_label="Unknown_climate")
    climate_major = climate.str[0].replace({"U": "Unknown"})
    region = continent + "_" + climate_major
    region = region.str.replace(r"\s+", "_", regex=True)
    return region


def summarize_regions_by_unique_sites(region: pd.Series, site_id: pd.Series, extra_cols: pd.DataFrame = None):
    """
    Summarize regions by sample count and unique coordinate-site count.
    """
    df = pd.DataFrame({
        "Region": region.astype(str).values,
        "Site_ID": site_id.astype(str).values,
    })

    if extra_cols is not None:
        extra = extra_cols.reset_index(drop=True).copy()
        df = pd.concat([df.reset_index(drop=True), extra], axis=1)

    agg_dict = {
        "n_samples": ("Site_ID", "size"),
        "n_sites": ("Site_ID", "nunique"),
    }

    if LAT_COL in df.columns:
        agg_dict["lat_min"] = (LAT_COL, "min")
        agg_dict["lat_max"] = (LAT_COL, "max")
    if LON_COL in df.columns:
        agg_dict["lon_min"] = (LON_COL, "min")
        agg_dict["lon_max"] = (LON_COL, "max")

    return (
        df.groupby("Region", as_index=False)
        .agg(**agg_dict)
        .sort_values(["n_sites", "n_samples"], ascending=[False, False])
    )


def lonlat_to_spherical_xyz(lat_values, lon_values):
    """
    Convert latitude/longitude in degrees to 3D unit-sphere coordinates.

    This avoids treating raw longitude/latitude as a flat Cartesian plane and is
    more appropriate for global data and KMeans-based spatial subdivision.
    """
    lat_rad = np.deg2rad(np.asarray(lat_values, dtype=float))
    lon_norm = normalize_longitude(lon_values)
    lon_rad = np.deg2rad(np.asarray(lon_norm, dtype=float))

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack([x, y, z])


def split_large_regions_by_internal_kmeans(
    base_region: pd.Series,
    site_id: pd.Series,
    lat_values,
    lon_values,
    max_unique_sites: int,
):
    """
    Split very large geo-climatic regions into internal KMeans spatial subregions.

    Logic
    -----
    1. Count unique Site_IDs in each base geo-climatic region.
    2. If a region has > max_unique_sites, split it into
       ceil(n_sites / max_unique_sites) KMeans clusters.
    3. KMeans is fitted only within that large region and uses site-level
       spherical coordinates derived from latitude and longitude.
    4. All samples from the same Site_ID stay in the same subregion.

    Why this is preferable to longitude-only splitting
    --------------------------------------------------
    Longitude quantile splitting can create artificial east-west bands and may
    ignore north-south structure. Internal KMeans considers both latitude and
    longitude and produces more spatially coherent subregions while preserving
    the geo-climatic region definition.
    """
    base_region = base_region.astype(str).reset_index(drop=True)
    site_id = site_id.astype(str).reset_index(drop=True)
    lat_series = pd.Series(pd.to_numeric(pd.Series(lat_values), errors="coerce")).reset_index(drop=True)
    lon_series = pd.Series(normalize_longitude(pd.to_numeric(pd.Series(lon_values), errors="coerce"))).reset_index(drop=True)

    sample_table = pd.DataFrame({
        "Base_region": base_region,
        "Site_ID": site_id,
        "Latitude": lat_series,
        "Longitude_normalized": lon_series,
    })

    site_table = (
        sample_table
        .groupby(["Base_region", "Site_ID"], as_index=False)
        .agg(
            Latitude=("Latitude", "median"),
            Longitude_normalized=("Longitude_normalized", "median"),
        )
    )

    split_records = []
    subregion_records = []
    site_to_split_region = {}

    for region_name, sub in site_table.groupby("Base_region", sort=True):
        sub = sub.copy().reset_index(drop=True)
        n_sites = len(sub)

        if SPLIT_LARGE_REGIONS and n_sites > max_unique_sites:
            n_subregions = int(np.ceil(n_sites / max_unique_sites))

            coords_xyz = lonlat_to_spherical_xyz(
                sub["Latitude"].values,
                sub["Longitude_normalized"].values,
            )

            kmeans = KMeans(
                n_clusters=n_subregions,
                random_state=RANDOM_STATE,
                n_init=50,
            )
            cluster_labels = kmeans.fit_predict(coords_xyz) + 1

            # Rename clusters by median longitude, so suffix order is spatially stable.
            sub["raw_cluster"] = cluster_labels
            cluster_order = (
                sub.groupby("raw_cluster")
                .agg(
                    median_lon=("Longitude_normalized", "median"),
                    median_lat=("Latitude", "median"),
                    n_sites=("Site_ID", "size"),
                )
                .sort_values(["median_lon", "median_lat"])
                .reset_index()
            )
            cluster_order["ordered_cluster"] = np.arange(1, len(cluster_order) + 1)
            raw_to_ordered = dict(zip(cluster_order["raw_cluster"], cluster_order["ordered_cluster"]))
            sub["subregion_id"] = sub["raw_cluster"].map(raw_to_ordered).astype(int)
            sub["Split_region"] = str(region_name) + "_km" + sub["subregion_id"].astype(str)
            split_flag = True

            for _, row in cluster_order.iterrows():
                ordered_id = int(row["ordered_cluster"])
                sub_sub = sub[sub["subregion_id"] == ordered_id]
                subregion_records.append({
                    "Base_region": region_name,
                    "Split_region": f"{region_name}_km{ordered_id}",
                    "n_sites": int(sub_sub["Site_ID"].nunique()),
                    "lat_min": float(sub_sub["Latitude"].min()),
                    "lat_max": float(sub_sub["Latitude"].max()),
                    "lon_min": float(sub_sub["Longitude_normalized"].min()),
                    "lon_max": float(sub_sub["Longitude_normalized"].max()),
                    "median_lat": float(sub_sub["Latitude"].median()),
                    "median_lon": float(sub_sub["Longitude_normalized"].median()),
                })
        else:
            n_subregions = 1
            sub["subregion_id"] = 1
            sub["Split_region"] = str(region_name)
            split_flag = False
            subregion_records.append({
                "Base_region": region_name,
                "Split_region": str(region_name),
                "n_sites": int(n_sites),
                "lat_min": float(sub["Latitude"].min()),
                "lat_max": float(sub["Latitude"].max()),
                "lon_min": float(sub["Longitude_normalized"].min()),
                "lon_max": float(sub["Longitude_normalized"].max()),
                "median_lat": float(sub["Latitude"].median()),
                "median_lon": float(sub["Longitude_normalized"].median()),
            })

        for _, row in sub.iterrows():
            site_to_split_region[row["Site_ID"]] = row["Split_region"]

        split_records.append({
            "Base_region": region_name,
            "n_sites_before_split": int(n_sites),
            "n_subregions": int(n_subregions),
            "was_split": bool(split_flag),
            "max_unique_sites_threshold": int(max_unique_sites),
            "split_method": LARGE_REGION_SPLIT_METHOD,
        })

    split_region = site_id.map(site_to_split_region)
    split_summary = pd.DataFrame(split_records).sort_values(
        ["was_split", "n_sites_before_split"], ascending=[False, False]
    )
    subregion_summary = pd.DataFrame(subregion_records).sort_values(
        ["Base_region", "Split_region"]
    )

    return split_region.astype(str), split_summary, subregion_summary

def merge_small_regions_by_unique_sites(region: pd.Series, site_id: pd.Series, min_unique_sites: int):
    """
    Merge regions with fewer than min_unique_sites unique sites into SMALL_REGION_LABEL.

    This should be applied after oversized regions have been split.

    Returns
    -------
    merged_region : pd.Series
    split_region_summary : pd.DataFrame
    """
    region_summary = summarize_regions_by_unique_sites(region, site_id)
    region_summary = region_summary.rename(columns={"Region": "Split_region"})

    small_regions = set(
        region_summary.loc[
            region_summary["n_sites"] < min_unique_sites,
            "Split_region"
        ].astype(str)
    )

    merged = region.astype(str).copy().reset_index(drop=True)
    merged.loc[merged.isin(small_regions)] = SMALL_REGION_LABEL
    return merged, region_summary


def add_region_and_site_columns(data_t: pd.DataFrame):
    """
    Add Site_ID, base geo-climatic region, split region, and final merged region columns.

    Region construction order
    -------------------------
    1. Base region = cleaned Continent + '_' + major Köppen climate class.
    2. Split base regions with > MAX_REGION_UNIQUE_SITES unique sites using internal KMeans spatial subregions.
    3. Merge split regions with < MIN_REGION_UNIQUE_SITES unique sites into Other_small_regions.
    """
    data_t = data_t.copy()
    data_t["Site_ID"] = make_site_id(data_t)
    data_t["Original_GeoClimate_Region"] = make_geoclimatic_region(data_t)

    original_region_summary = summarize_regions_by_unique_sites(
        data_t["Original_GeoClimate_Region"],
        data_t["Site_ID"],
        extra_cols=data_t[[LAT_COL, LON_COL]],
    ).rename(columns={"Region": "Original_region"})

    split_region, large_region_split_summary, kmeans_subregion_summary = split_large_regions_by_internal_kmeans(
        data_t["Original_GeoClimate_Region"],
        data_t["Site_ID"],
        data_t[LAT_COL].values,
        data_t[LON_COL].values,
        max_unique_sites=MAX_REGION_UNIQUE_SITES,
    )
    data_t["Split_GeoClimate_Region"] = split_region.values

    merged_region, split_region_summary = merge_small_regions_by_unique_sites(
        data_t["Split_GeoClimate_Region"],
        data_t["Site_ID"],
        min_unique_sites=MIN_REGION_UNIQUE_SITES,
    )
    data_t[REGION_COL_OUT] = merged_region.values

    final_region_summary = summarize_regions_by_unique_sites(
        data_t[REGION_COL_OUT],
        data_t["Site_ID"],
        extra_cols=data_t[[LAT_COL, LON_COL]],
    ).rename(columns={"Region": REGION_COL_OUT})

    diagnostics = {
        "original_region_summary": original_region_summary,
        "large_region_split_summary": large_region_split_summary,
        "kmeans_subregion_summary": kmeans_subregion_summary,
        "split_region_summary": split_region_summary,
        "final_region_summary": final_region_summary,
    }
    return data_t, diagnostics

def site_level_metric_from_oof(oof_df: pd.DataFrame, pred_col: str):
    """
    Calculate site-balanced metrics by averaging observed and predicted values by Site_ID.
    """
    valid = oof_df[["Site_ID", "Observed", pred_col]].dropna().copy()
    if valid.empty:
        return {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan}, pd.DataFrame()

    site_df = (
        valid.groupby("Site_ID", as_index=False)
        .agg(
            Observed_site_mean=("Observed", "mean"),
            Predicted_site_mean=(pred_col, "mean"),
            n_samples=("Observed", "size"),
        )
    )
    metrics = calc_metrics(site_df["Observed_site_mean"].values, site_df["Predicted_site_mean"].values)
    return metrics, site_df


def plot_observed_vs_predicted(y_true, y_pred, metrics, target_name, output_path, title_prefix, xlabel):
    """Save observed-vs-predicted scatter plot."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) < 2:
        return

    plt.figure(figsize=(7, 6))
    plt.scatter(
        y_pred,
        y_true,
        alpha=0.65,
        s=45,
        edgecolors="white",
        linewidths=0.4,
    )

    min_val = min(np.nanmin(y_true), np.nanmin(y_pred))
    max_val = max(np.nanmax(y_true), np.nanmax(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2, label="1:1 line")

    textstr = "\n".join([
        "Algorithm: Random Forest",
        f"R² = {metrics['R2']:.3f}",
        f"RMSE = {metrics['RMSE']:.3f}",
        f"MAE = {metrics['MAE']:.3f}",
    ])

    props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
    plt.gca().text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
    )

    plt.title(f"{title_prefix} - {target_name}", fontsize=14, fontweight="bold")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Observed value", fontsize=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_loro_cv_for_target(
    X: pd.DataFrame,
    y_raw: np.ndarray,
    data_t: pd.DataFrame,
    target_name: str,
    rf_params: dict,
):
    """Run site-balanced geo-climatic Leave-One-Region-Out CV for one target variable."""
    region_labels = data_t[REGION_COL_OUT].astype(str).values
    unique_regions = list(pd.Series(region_labels).dropna().sort_values().unique())

    if len(unique_regions) < 2:
        raise ValueError(f"{target_name}: fewer than two regions available for LORO-CV.")

    cv_name = "LORO_GeoClimate_SiteBalanced_CV"

    oof_pred = np.full(len(y_raw), np.nan, dtype=float)
    fold_records = []
    importance_records = []

    for fold_idx, held_out_region in enumerate(unique_regions, start=1):
        test_idx = np.where(region_labels == held_out_region)[0]
        train_idx = np.where(region_labels != held_out_region)[0]

        if len(test_idx) < 2 or len(train_idx) < 10:
            print(
                f"   - Skipping region={held_out_region}: "
                f"N_test={len(test_idx)}, N_train={len(train_idx)}"
            )
            continue

        model = make_rf_pipeline(rf_params)
        model.fit(X.iloc[train_idx], y_raw[train_idx])

        pred = model.predict(X.iloc[test_idx])
        oof_pred[test_idx] = pred

        sample_metric = calc_metrics(y_raw[test_idx], pred)

        fold_oof_df = pd.DataFrame({
            "Site_ID": data_t.iloc[test_idx]["Site_ID"].values,
            "Observed": y_raw[test_idx],
            "Predicted_LORO_OOF": pred,
        })
        site_metric, fold_site_df = site_level_metric_from_oof(fold_oof_df, "Predicted_LORO_OOF")

        n_test_sites = int(fold_oof_df["Site_ID"].nunique())
        small_test_warning = (len(test_idx) < MIN_TEST_SAMPLES_WARN) or (n_test_sites < MIN_TEST_SITES_WARN)

        fold_records.append({
            "Target Variable": target_name,
            "CV": cv_name,
            "Fold": fold_idx,
            "Held_out_region": held_out_region,
            "Sample_R2": sample_metric["R2"],
            "Sample_RMSE": sample_metric["RMSE"],
            "Sample_MAE": sample_metric["MAE"],
            "SiteBalanced_R2": site_metric["R2"],
            "SiteBalanced_RMSE": site_metric["RMSE"],
            "SiteBalanced_MAE": site_metric["MAE"],
            "N_train_samples": int(len(train_idx)),
            "N_test_samples": int(len(test_idx)),
            "N_train_sites": int(data_t.iloc[train_idx]["Site_ID"].nunique()),
            "N_test_sites": n_test_sites,
            "Small_test_fold_warning": bool(small_test_warning),
            "y_test_mean": float(np.mean(y_raw[test_idx])),
            "y_test_sd": float(np.std(y_raw[test_idx], ddof=1)) if len(test_idx) > 1 else np.nan,
            "y_test_min": float(np.min(y_raw[test_idx])),
            "y_test_max": float(np.max(y_raw[test_idx])),
        })

        rf_model = model.named_steps["randomforest"]
        for feature, importance in zip(X.columns, rf_model.feature_importances_):
            importance_records.append({
                "Target Variable": target_name,
                "CV": cv_name,
                "Fold": fold_idx,
                "Held_out_region": held_out_region,
                "Feature": feature,
                "Importance": float(importance),
            })

        warn_text = " [small fold]" if small_test_warning else ""
        print(
            f"   - {cv_name} fold {fold_idx}, held-out region={held_out_region}: "
            f"sample_R2={sample_metric['R2']:.4f}, "
            f"site_R2={site_metric['R2']:.4f}, "
            f"sample_RMSE={sample_metric['RMSE']:.4f}, "
            f"site_RMSE={site_metric['RMSE']:.4f}, "
            f"N_test_samples={len(test_idx)}, N_test_sites={n_test_sites}{warn_text}"
        )

    valid_oof = ~np.isnan(oof_pred)
    if valid_oof.sum() < 2:
        raise ValueError(f"{target_name}: no valid LORO predictions generated.")

    sample_overall_metrics = calc_metrics(y_raw[valid_oof], oof_pred[valid_oof])

    fold_df = pd.DataFrame(fold_records)
    importance_df = pd.DataFrame(importance_records)

    oof_df = pd.DataFrame({
        "Original_Index": data_t.index.values,
        "Target Variable": target_name,
        "Observed": y_raw,
        "Predicted_LORO_OOF": oof_pred,
        "Residual": y_raw - oof_pred,
        "Site_ID": data_t["Site_ID"].values,
        REGION_COL_OUT: data_t[REGION_COL_OUT].values,
        "Split_GeoClimate_Region": data_t["Split_GeoClimate_Region"].values,
        "Original_GeoClimate_Region": data_t["Original_GeoClimate_Region"].values,
        CONTINENT_COL: data_t[CONTINENT_COL].values,
        CLIMATE_COL: data_t[CLIMATE_COL].values,
        LAT_COL: data_t[LAT_COL].values,
        LON_COL: data_t[LON_COL].values,
    })

    id_col = find_id_col(data_t)
    if id_col is not None:
        oof_df.insert(0, id_col, data_t[id_col].values)

    site_overall_metrics, site_oof_df = site_level_metric_from_oof(oof_df, "Predicted_LORO_OOF")

    # Add region metadata to site-level OOF table.
    site_meta = (
        oof_df[["Site_ID", REGION_COL_OUT, CONTINENT_COL, CLIMATE_COL, LAT_COL, LON_COL]]
        .drop_duplicates(subset=["Site_ID"])
    )
    if not site_oof_df.empty:
        site_oof_df = site_oof_df.merge(site_meta, on="Site_ID", how="left")
        site_oof_df["Residual_site_mean"] = (
            site_oof_df["Observed_site_mean"] - site_oof_df["Predicted_site_mean"]
        )

    region_summary = (
        oof_df.groupby(REGION_COL_OUT, as_index=False)
        .agg(
            n_samples=("Observed", "size"),
            n_sites=("Site_ID", "nunique"),
            y_mean=("Observed", "mean"),
            y_sd=("Observed", "std"),
            y_min=("Observed", "min"),
            y_max=("Observed", "max"),
            lat_min=(LAT_COL, "min"),
            lat_max=(LAT_COL, "max"),
            lon_min=(LON_COL, "min"),
            lon_max=(LON_COL, "max"),
        )
        .sort_values(["n_sites", "n_samples"], ascending=[False, False])
    )

    summary_record = {
        "Target Variable": target_name,
        "Model": "Random Forest",
        "CV": cv_name,
        "N_samples": int(valid_oof.sum()),
        "N_sites": int(oof_df.loc[valid_oof, "Site_ID"].nunique()),
        "N_regions": int(region_summary.shape[0]),
        "Min_region_unique_sites_threshold": MIN_REGION_UNIQUE_SITES,
        "Max_region_unique_sites_split_threshold": MAX_REGION_UNIQUE_SITES,
        "Split_large_regions": SPLIT_LARGE_REGIONS,
        "Large_region_split_method": LARGE_REGION_SPLIT_METHOD,
        "Min_region_sites_after_merging": int(region_summary["n_sites"].min()),
        "Max_region_sites_after_merging": int(region_summary["n_sites"].max()),
        "Min_region_samples_after_merging": int(region_summary["n_samples"].min()),
        "Max_region_samples_after_merging": int(region_summary["n_samples"].max()),
        "Any_small_test_fold_warning": bool(fold_df["Small_test_fold_warning"].any()) if not fold_df.empty else np.nan,
        "SampleLevel_LORO_OOF_R2": sample_overall_metrics["R2"],
        "SampleLevel_LORO_OOF_RMSE": sample_overall_metrics["RMSE"],
        "SampleLevel_LORO_OOF_MAE": sample_overall_metrics["MAE"],
        "SiteBalanced_LORO_OOF_R2": site_overall_metrics["R2"],
        "SiteBalanced_LORO_OOF_RMSE": site_overall_metrics["RMSE"],
        "SiteBalanced_LORO_OOF_MAE": site_overall_metrics["MAE"],
        "Fold_Sample_R2_mean": float(fold_df["Sample_R2"].mean()) if not fold_df.empty else np.nan,
        "Fold_Sample_R2_sd": float(fold_df["Sample_R2"].std()) if not fold_df.empty else np.nan,
        "Fold_SiteBalanced_R2_mean": float(fold_df["SiteBalanced_R2"].mean()) if not fold_df.empty else np.nan,
        "Fold_SiteBalanced_R2_sd": float(fold_df["SiteBalanced_R2"].std()) if not fold_df.empty else np.nan,
        "Fold_N_test_samples_min": int(fold_df["N_test_samples"].min()) if not fold_df.empty else np.nan,
        "Fold_N_test_samples_max": int(fold_df["N_test_samples"].max()) if not fold_df.empty else np.nan,
        "Fold_N_test_sites_min": int(fold_df["N_test_sites"].min()) if not fold_df.empty else np.nan,
        "Fold_N_test_sites_max": int(fold_df["N_test_sites"].max()) if not fold_df.empty else np.nan,
        "RF_params_used": str(rf_params),
    }

    return (
        summary_record,
        fold_df,
        oof_df,
        site_oof_df,
        importance_df,
        region_summary,
        sample_overall_metrics,
        site_overall_metrics,
        cv_name,
    )


# =============================================================================
# 3. Main workflow
# =============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    print("=" * 100)
    print("Loading original data and previous Random Forest results")
    print("=" * 100)

    data = pd.read_excel(DATA_PATH, sheet_name=DATA_SHEET_NAME)
    print(f"Original data: {DATA_PATH}")
    print(f"Data shape: {data.shape}")

    check_required_columns(
        data,
        FACTOR_COLS + [LAT_COL, LON_COL, CONTINENT_COL, CLIMATE_COL],
        "Original data",
    )

    # Clean region-related columns once globally.
    data[CONTINENT_COL] = clean_text_series(data[CONTINENT_COL], missing_label="Unknown_continent")
    data[CLIMATE_COL] = clean_text_series(data[CLIMATE_COL], missing_label="Unknown_climate")

    previous_results = pd.read_excel(PREVIOUS_RESULTS_EXCEL, sheet_name=PREVIOUS_RESULTS_SHEET)
    print(f"Previous results: {PREVIOUS_RESULTS_EXCEL} / sheet: {PREVIOUS_RESULTS_SHEET}")
    print(f"Previous results shape: {previous_results.shape}")

    check_required_columns(
        previous_results,
        ["Target Variable", "Model", "Best Parameters"],
        "Previous Model Results sheet",
    )

    rf_results = previous_results[previous_results["Model"].astype(str) == "Random Forest"].copy()
    if rf_results.empty:
        raise ValueError("No rows with Model == 'Random Forest' were found in the previous results.")

    # If duplicated target-model rows exist, keep the one with the highest R2_test when available.
    if "R2_test" in rf_results.columns:
        rf_results = (
            rf_results
            .sort_values(["Target Variable", "R2_test"], ascending=[True, False])
            .groupby("Target Variable", as_index=False)
            .head(1)
        )
    else:
        rf_results = rf_results.drop_duplicates(subset=["Target Variable"], keep="first")

    if TARGET_COLS is None:
        target_cols = list(rf_results["Target Variable"].astype(str).unique())
    else:
        target_cols = TARGET_COLS
        rf_results = rf_results[rf_results["Target Variable"].isin(target_cols)].copy()

    missing_targets = [t for t in target_cols if t not in data.columns]
    if missing_targets:
        raise KeyError("Original data is missing target columns:\n" + "\n".join(missing_targets))

    print(f"Targets selected from Random Forest results ({len(target_cols)}): {target_cols}")
    print(
        "LORO-CV: base region = cleaned Continent + major Köppen climate class; "
        f"regions with > {MAX_REGION_UNIQUE_SITES} unique sites are split by internal KMeans; "
        f"regions with < {MIN_REGION_UNIQUE_SITES} unique sites are merged into {SMALL_REGION_LABEL}."
    )
    print("Main reporting metric: SiteBalanced_LORO_OOF_R2 / RMSE / MAE")

    all_summary = []
    all_folds = []
    all_oof = []
    all_site_oof = []
    all_importance = []
    all_region_summary = []
    all_original_region_summary = []
    all_large_region_split_summary = []
    all_kmeans_subregion_summary = []
    all_split_region_summary = []
    all_final_region_summary_pre_cv = []
    params_used = []

    id_col = find_id_col(data)

    for target_name in target_cols:
        print("\n" + "=" * 100)
        print(f"Leave-One-Region-Out CV for target: {target_name}")
        print("=" * 100)

        result_row = rf_results[rf_results["Target Variable"].astype(str) == str(target_name)]
        if result_row.empty:
            print(f"Skipping {target_name}: no Random Forest parameter row found in previous results.")
            continue

        best_params_string = result_row.iloc[0]["Best Parameters"]
        rf_params = parse_random_forest_params(best_params_string)

        print(f"RF parameters from previous results: {rf_params}")

        y_all = pd.to_numeric(data[target_name], errors="coerce")

        required_cols = FACTOR_COLS + [target_name, LAT_COL, LON_COL, CONTINENT_COL, CLIMATE_COL]
        if id_col is not None:
            required_cols = [id_col] + required_cols

        data_t = data[required_cols].copy()
        data_t[target_name] = y_all
        data_t[LAT_COL] = pd.to_numeric(data_t[LAT_COL], errors="coerce")
        data_t[LON_COL] = pd.to_numeric(data_t[LON_COL], errors="coerce")

        # Drop missing target and coordinates/region labels. Missing predictors are handled by imputation.
        data_t = data_t.dropna(subset=[target_name, LAT_COL, LON_COL, CONTINENT_COL, CLIMATE_COL])

        if len(data_t) < 20:
            print(f"Skipping {target_name}: too few valid samples after filtering ({len(data_t)}).")
            continue

        data_t, region_diagnostics = add_region_and_site_columns(data_t)
        original_region_summary = region_diagnostics["original_region_summary"]
        large_region_split_summary = region_diagnostics["large_region_split_summary"]
        kmeans_subregion_summary = region_diagnostics["kmeans_subregion_summary"]
        split_region_summary = region_diagnostics["split_region_summary"]
        final_region_summary_pre_cv = region_diagnostics["final_region_summary"]

        if data_t[REGION_COL_OUT].nunique() < 2:
            print(f"Skipping {target_name}: fewer than two merged regions.")
            continue

        X = data_t[FACTOR_COLS].apply(pd.to_numeric, errors="coerce")
        y = data_t[target_name].values.astype(float)

        (
            summary_record,
            fold_df,
            oof_df,
            site_oof_df,
            importance_df,
            region_summary,
            sample_metrics,
            site_metrics,
            cv_name,
        ) = run_loro_cv_for_target(
            X=X,
            y_raw=y,
            data_t=data_t,
            target_name=target_name,
            rf_params=rf_params,
        )

        # Add previous random-test/CV metrics when available for comparison.
        result_row_dict = result_row.iloc[0].to_dict()
        for col in [
            "CV_R2_mean_best", "CV_R2_sd_best", "R2_train", "R2_test",
            "RMSE_test", "MAE_test", "R2_overall"
        ]:
            if col in result_row_dict:
                summary_record[f"Previous_{col}"] = result_row_dict[col]

        all_summary.append(summary_record)
        all_folds.append(fold_df)
        all_oof.append(oof_df)
        all_site_oof.append(site_oof_df)
        all_importance.append(importance_df)

        region_summary.insert(0, "Target Variable", target_name)
        all_region_summary.append(region_summary)

        original_region_summary.insert(0, "Target Variable", target_name)
        all_original_region_summary.append(original_region_summary)

        large_region_split_summary.insert(0, "Target Variable", target_name)
        all_large_region_split_summary.append(large_region_split_summary)

        kmeans_subregion_summary.insert(0, "Target Variable", target_name)
        all_kmeans_subregion_summary.append(kmeans_subregion_summary)

        split_region_summary.insert(0, "Target Variable", target_name)
        all_split_region_summary.append(split_region_summary)

        final_region_summary_pre_cv.insert(0, "Target Variable", target_name)
        all_final_region_summary_pre_cv.append(final_region_summary_pre_cv)

        params_used.append({
            "Target Variable": target_name,
            "Best Parameters Original": str(best_params_string),
            "RF Parameters Parsed": str(rf_params),
            "CV": cv_name,
        })

        # Save per-target CSV files.
        prefix = safe_filename(target_name)
        fold_df.to_csv(
            os.path.join(OUTPUT_DIR, f"{prefix}_LORO_fold_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        oof_df.to_csv(
            os.path.join(OUTPUT_DIR, f"{prefix}_LORO_sample_oof_predictions.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        site_oof_df.to_csv(
            os.path.join(OUTPUT_DIR, f"{prefix}_LORO_sitebalanced_oof_predictions.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        importance_df.to_csv(
            os.path.join(OUTPUT_DIR, f"{prefix}_LORO_feature_importance_by_fold.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        region_summary.to_csv(
            os.path.join(OUTPUT_DIR, f"{prefix}_LORO_region_summary_after_merging.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        original_region_summary.to_csv(
            os.path.join(OUTPUT_DIR, f"{prefix}_LORO_original_region_summary_before_splitting.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        large_region_split_summary.to_csv(
            os.path.join(OUTPUT_DIR, f"{prefix}_LORO_large_region_split_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        kmeans_subregion_summary.to_csv(
            os.path.join(OUTPUT_DIR, f"{prefix}_LORO_kmeans_subregion_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        split_region_summary.to_csv(
            os.path.join(OUTPUT_DIR, f"{prefix}_LORO_split_region_summary_before_small_merge.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        final_region_summary_pre_cv.to_csv(
            os.path.join(OUTPUT_DIR, f"{prefix}_LORO_final_region_summary_after_split_and_merge.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        plot_observed_vs_predicted(
            y_true=oof_df["Observed"].values,
            y_pred=oof_df["Predicted_LORO_OOF"].values,
            metrics=sample_metrics,
            target_name=target_name,
            output_path=os.path.join(OUTPUT_DIR, f"{prefix}_LORO_sample_level_validation.pdf"),
            title_prefix="Sample-level LORO-CV",
            xlabel="Predicted value (sample-level LORO OOF)",
        )

        if not site_oof_df.empty:
            plot_observed_vs_predicted(
                y_true=site_oof_df["Observed_site_mean"].values,
                y_pred=site_oof_df["Predicted_site_mean"].values,
                metrics=site_metrics,
                target_name=target_name,
                output_path=os.path.join(OUTPUT_DIR, f"{prefix}_LORO_sitebalanced_validation.pdf"),
                title_prefix="Site-balanced LORO-CV",
                xlabel="Predicted value (site-mean LORO OOF)",
            )

        # Fit and save final RF model on all valid samples for this target using the same RF params.
        final_model = make_rf_pipeline(rf_params)
        final_model.fit(X, y)
        final_model_file = os.path.join(
            MODEL_SAVE_PATH,
            f"{prefix}_RandomForest_final_model_from_previous_best_params.joblib",
        )
        dump({
            "model": final_model,
            "feature_columns": FACTOR_COLS,
            "target": target_name,
            "rf_params": rf_params,
            "loro_cv_summary": summary_record,
            "region_definition": "cleaned Continent + major Köppen climate class; oversized regions split by internal KMeans using spherical coordinates; small regions merged",
            "min_region_unique_sites": MIN_REGION_UNIQUE_SITES,
            "max_region_unique_sites_split_threshold": MAX_REGION_UNIQUE_SITES,
            "large_region_split_method": LARGE_REGION_SPLIT_METHOD,
            "site_coord_decimals": SITE_COORD_DECIMALS,
        }, final_model_file)

        print(
            f"LORO OOF for {target_name}: "
            f"sample_R2={sample_metrics['R2']:.4f}, sample_RMSE={sample_metrics['RMSE']:.4f}, "
            f"site_R2={site_metrics['R2']:.4f}, site_RMSE={site_metrics['RMSE']:.4f}"
        )
        print(f"Final model saved: {final_model_file}")

    # -------------------------------------------------------------------------
    # Save combined outputs.
    # -------------------------------------------------------------------------
    summary_df = pd.DataFrame(all_summary)
    folds_df = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()
    oof_all_df = pd.concat(all_oof, ignore_index=True) if all_oof else pd.DataFrame()
    site_oof_all_df = pd.concat(all_site_oof, ignore_index=True) if all_site_oof else pd.DataFrame()
    importance_all_df = pd.concat(all_importance, ignore_index=True) if all_importance else pd.DataFrame()
    region_summary_all_df = pd.concat(all_region_summary, ignore_index=True) if all_region_summary else pd.DataFrame()
    original_region_summary_all_df = pd.concat(all_original_region_summary, ignore_index=True) if all_original_region_summary else pd.DataFrame()
    large_region_split_summary_all_df = pd.concat(all_large_region_split_summary, ignore_index=True) if all_large_region_split_summary else pd.DataFrame()
    kmeans_subregion_summary_all_df = pd.concat(all_kmeans_subregion_summary, ignore_index=True) if all_kmeans_subregion_summary else pd.DataFrame()
    split_region_summary_all_df = pd.concat(all_split_region_summary, ignore_index=True) if all_split_region_summary else pd.DataFrame()
    final_region_summary_pre_cv_all_df = pd.concat(all_final_region_summary_pre_cv, ignore_index=True) if all_final_region_summary_pre_cv else pd.DataFrame()
    params_used_df = pd.DataFrame(params_used)

    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "RF_LORO_KMeansLargeRegion_sitebalanced_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    folds_df.to_csv(
        os.path.join(OUTPUT_DIR, "RF_LORO_KMeansLargeRegion_fold_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    oof_all_df.to_csv(
        os.path.join(OUTPUT_DIR, "RF_LORO_KMeansLargeRegion_sample_oof_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    site_oof_all_df.to_csv(
        os.path.join(OUTPUT_DIR, "RF_LORO_KMeansLargeRegion_sitebalanced_oof_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    importance_all_df.to_csv(
        os.path.join(OUTPUT_DIR, "RF_LORO_KMeansLargeRegion_feature_importance_by_fold.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    region_summary_all_df.to_csv(
        os.path.join(OUTPUT_DIR, "RF_LORO_KMeansLargeRegion_region_summary_after_split_and_merge.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    original_region_summary_all_df.to_csv(
        os.path.join(OUTPUT_DIR, "RF_LORO_KMeansLargeRegion_original_region_summary_before_splitting.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    large_region_split_summary_all_df.to_csv(
        os.path.join(OUTPUT_DIR, "RF_LORO_KMeansLargeRegion_large_region_split_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    kmeans_subregion_summary_all_df.to_csv(
        os.path.join(OUTPUT_DIR, "RF_LORO_KMeansLargeRegion_kmeans_subregion_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    split_region_summary_all_df.to_csv(
        os.path.join(OUTPUT_DIR, "RF_LORO_KMeansLargeRegion_split_region_summary_before_small_merge.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    final_region_summary_pre_cv_all_df.to_csv(
        os.path.join(OUTPUT_DIR, "RF_LORO_KMeansLargeRegion_final_region_summary_after_split_and_merge.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    params_used_df.to_csv(
        os.path.join(OUTPUT_DIR, "RF_parameters_used_from_previous_results.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    excel_path = os.path.join(OUTPUT_DIR, "RF_LORO_KMeansLargeRegion_sitebalanced_results.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        summary_df.to_excel(writer, sheet_name="LORO Summary", index=False)
        folds_df.to_excel(writer, sheet_name="Fold Metrics", index=False)
        oof_all_df.to_excel(writer, sheet_name="Sample OOF", index=False)
        site_oof_all_df.to_excel(writer, sheet_name="Site-balanced OOF", index=False)
        importance_all_df.to_excel(writer, sheet_name="Feature Importance", index=False)
        region_summary_all_df.to_excel(writer, sheet_name="Merged Region Summary", index=False)
        original_region_summary_all_df.to_excel(writer, sheet_name="Original Region Summary", index=False)
        large_region_split_summary_all_df.to_excel(writer, sheet_name="Large Region Splits", index=False)
        kmeans_subregion_summary_all_df.to_excel(writer, sheet_name="KMeans Subregions", index=False)
        split_region_summary_all_df.to_excel(writer, sheet_name="Split Region Summary", index=False)
        final_region_summary_pre_cv_all_df.to_excel(writer, sheet_name="Final Region Summary", index=False)
        params_used_df.to_excel(writer, sheet_name="RF Params Used", index=False)

    print("\n" + "=" * 100)
    print("Finished Random Forest site-balanced internal-KMeans-split-large-region Leave-One-Region-Out CV")
    print("=" * 100)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Excel summary: {excel_path}")
    print(f"Final RF models: {MODEL_SAVE_PATH}")

    if not summary_df.empty:
        cols_to_show = [
            "Target Variable",
            "SampleLevel_LORO_OOF_R2",
            "SiteBalanced_LORO_OOF_R2",
            "SiteBalanced_LORO_OOF_RMSE",
            "SiteBalanced_LORO_OOF_MAE",
            "Fold_N_test_samples_min",
            "Fold_N_test_samples_max",
            "Fold_N_test_sites_min",
            "Fold_N_test_sites_max",
            "N_regions",
            "N_sites",
            "N_samples",
            "Any_small_test_fold_warning",
        ]
        print("\nLORO-CV summary:")
        print(summary_df[cols_to_show].to_string(index=False))


if __name__ == "__main__":
    main()
