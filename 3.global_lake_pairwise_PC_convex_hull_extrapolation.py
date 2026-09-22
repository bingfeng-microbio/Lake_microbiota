# -*- coding: utf-8 -*-
"""
Global lake environmental extrapolation analysis.

Core workflow:
1. Fit median imputation, standardization, and PCA using the training data.
2. Retain PCs explaining >=90% cumulative variance.
3. Build a 2-D convex hull for every pair of retained PCs.
4. Evaluate every global lake-month from 2001-01 to 2020-12.
5. Calculate each lake's mean PCA extrapolation and fraction of supported months.
6. Retain lakes for which at least 50% of the 240 months are environmentally supported.

Only the main analysis table is saved.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay, QhullError
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. Configuration
# =============================================================================

BASE_DIR = Path("/Users/bingfengchen/Desktop/Freshwater")

TRAINING_FILE = BASE_DIR / "data_for_ML.xlsx"
TRAINING_SHEET = 0

MONTHLY_DIR = BASE_DIR / "global_lake_factors_2001_2020_by_month"
MONTHLY_FILE_TEMPLATE = "global_lake_factors_{year}-{month:02d}.csv"

OUTPUT_DIR = BASE_DIR / "PC_convex_hull_extrapolation"
OUTPUT_FILE = OUTPUT_DIR / "global_lake_environmental_extrapolation_core.csv"

START_YEAR = 2001
END_YEAR = 2020

PCA_CUMULATIVE_VARIANCE = 0.90
ENVIRONMENTAL_EXTRAPOLATION_CUTOFF = 0.05
MIN_SUPPORTED_MONTH_FRACTION = 0.50

CHUNK_SIZE = 25000

FACTOR_COLS = [
    "si10", "sp", "ssr", "t2m", "tp",
    "Lake_area", "Shore_dev", "Dis_avg", "Res_time", "Elevation",
    "Forest", "Grassland", "Barren", "Cropland", "Urban",
]

LAKE_ID_COL = "Hylak_id"
LAT_COL = "Latitude"
LON_COL = "Longitude"


# =============================================================================
# 2. Utilities
# =============================================================================

def require_columns(df, cols, name):
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def canonical_id_series(series):
    raw = series.copy()
    numeric = pd.to_numeric(raw, errors="coerce")
    out = raw.astype(str).str.strip()

    numeric_array = numeric.fillna(0).to_numpy(dtype=float)
    integer_like = numeric.notna() & np.isclose(
        numeric_array, np.round(numeric_array)
    )
    out.loc[integer_like] = (
        np.round(numeric.loc[integer_like]).astype("int64").astype(str)
    )

    return out.replace({
        "": np.nan,
        "nan": np.nan,
        "NaN": np.nan,
        "None": np.nan,
    })


def normalize_longitude(values):
    values = np.asarray(values, dtype=float)
    return ((values + 180.0) % 360.0) - 180.0


def resolve_monthly_file(year, month):
    plain = MONTHLY_DIR / MONTHLY_FILE_TEMPLATE.format(
        year=year,
        month=month,
    )
    gzip_file = plain.with_suffix(plain.suffix + ".gz")

    if plain.exists():
        return plain
    if gzip_file.exists():
        return gzip_file

    raise FileNotFoundError(
        f"Monthly predictor file not found:\n{plain}\nor\n{gzip_file}"
    )


# =============================================================================
# 3. Training data and PCA
# =============================================================================

def prepare_training_data(training):
    require_columns(
        training,
        FACTOR_COLS + [LAT_COL, LON_COL],
        "Training data",
    )

    work = training.copy()
    work[LAT_COL] = pd.to_numeric(work[LAT_COL], errors="coerce")
    work[LON_COL] = pd.to_numeric(work[LON_COL], errors="coerce")
    work = work.dropna(subset=[LAT_COL, LON_COL]).reset_index(drop=True)

    X_raw = work[FACTOR_COLS].apply(pd.to_numeric, errors="coerce")

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    pca = PCA(
        n_components=PCA_CUMULATIVE_VARIANCE,
        svd_solver="full",
    )
    X_pc = pca.fit_transform(X_scaled)

    return work, X_pc, imputer, scaler, pca


def build_pairwise_hulls(X_pc):
    pair_models = []

    for pc_a, pc_b in combinations(range(X_pc.shape[1]), 2):
        unique_points = np.unique(X_pc[:, [pc_a, pc_b]], axis=0)

        if len(unique_points) < 3:
            continue

        try:
            hull = ConvexHull(unique_points, qhull_options="QJ")
            hull_vertices = unique_points[hull.vertices]
            triangulation = Delaunay(
                hull_vertices,
                qhull_options="QJ",
            )
            pair_models.append((pc_a, pc_b, triangulation))
        except QhullError:
            continue

    if not pair_models:
        raise RuntimeError("No valid pairwise PC convex hulls were created.")

    return pair_models


def calculate_pca_extrapolation(X_pc_global, pair_models):
    inside_count = np.zeros(len(X_pc_global), dtype=np.uint16)

    for pc_a, pc_b, triangulation in pair_models:
        inside = (
            triangulation.find_simplex(
                X_pc_global[:, [pc_a, pc_b]]
            ) >= 0
        )
        inside_count += inside.astype(np.uint16)

    coverage = inside_count.astype(np.float32) / float(len(pair_models))
    return 1.0 - coverage


# =============================================================================
# 4. Global lake reference
# =============================================================================

def load_global_reference(first_file):
    first = pd.read_csv(
        first_file,
        usecols=[LAKE_ID_COL, LON_COL, LAT_COL],
        low_memory=False,
    )

    first[LAKE_ID_COL] = canonical_id_series(first[LAKE_ID_COL])

    if first[LAKE_ID_COL].isna().any():
        raise ValueError("Missing Hylak_id in first global file.")
    if first[LAKE_ID_COL].duplicated().any():
        raise ValueError("Duplicated Hylak_id in first global file.")

    first[LON_COL] = normalize_longitude(
        pd.to_numeric(first[LON_COL], errors="coerce")
    )
    first[LAT_COL] = pd.to_numeric(first[LAT_COL], errors="coerce")

    return first.reset_index(drop=True)




# =============================================================================
# 5. Monthly environmental support
# =============================================================================

def process_months(
    global_ref,
    imputer,
    scaler,
    pca,
    pair_models,
):
    months = [
        (year, month)
        for year in range(START_YEAR, END_YEAR + 1)
        for month in range(1, 13)
    ]

    n_lakes = len(global_ref)
    n_months = len(months)

    lake_index = pd.Index(
        global_ref[LAKE_ID_COL].astype(str),
        name=LAKE_ID_COL,
    )

    extrapolation_sum = np.zeros(n_lakes, dtype=np.float64)
    valid_month_count = np.zeros(n_lakes, dtype=np.uint16)
    supported_month_count = np.zeros(n_lakes, dtype=np.uint16)

    for month_number, (year, month) in enumerate(months, start=1):
        input_file = resolve_monthly_file(year, month)

        print(
            f"[{month_number:03d}/{n_months:03d}] "
            f"{year}-{month:02d}: {input_file.name}"
        )

        monthly = pd.read_csv(
            input_file,
            usecols=[LAKE_ID_COL] + FACTOR_COLS,
            low_memory=False,
        )
        monthly[LAKE_ID_COL] = canonical_id_series(
            monthly[LAKE_ID_COL]
        )

        if monthly[LAKE_ID_COL].duplicated().any():
            raise ValueError(
                f"Duplicated Hylak_id in {input_file.name}"
            )

        monthly = (
            monthly
            .set_index(LAKE_ID_COL)
            .reindex(lake_index)
        )

        X_raw = (
            monthly[FACTOR_COLS]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )

        # Keep the formal rule from the original analysis:
        # only complete predictor rows are evaluated.
        valid_indices = np.where(
            np.isfinite(X_raw).all(axis=1)
        )[0]

        for start in range(0, len(valid_indices), CHUNK_SIZE):
            selected = valid_indices[start:start + CHUNK_SIZE]
            X_chunk = X_raw[selected]

            X_imputed = imputer.transform(X_chunk)
            X_scaled = scaler.transform(X_imputed)
            X_pc_global = pca.transform(X_scaled)

            extrapolation = calculate_pca_extrapolation(
                X_pc_global,
                pair_models,
            )

            extrapolation_sum[selected] += extrapolation
            valid_month_count[selected] += 1
            supported_month_count[selected] += (
                extrapolation <= ENVIRONMENTAL_EXTRAPOLATION_CUTOFF
            ).astype(np.uint16)

    mean_extrapolation = np.divide(
        extrapolation_sum,
        valid_month_count,
        out=np.full(n_lakes, np.nan, dtype=float),
        where=valid_month_count > 0,
    )

    supported_fraction_expected = (
        supported_month_count.astype(float) / float(n_months)
    )

    return (
        n_months,
        valid_month_count,
        supported_month_count,
        mean_extrapolation,
        supported_fraction_expected,
    )


# =============================================================================
# 6. Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Global lake environmental extrapolation analysis")
    print("=" * 100)

    training = pd.read_excel(
        TRAINING_FILE,
        sheet_name=TRAINING_SHEET,
    )

    (
        training_work,
        X_pc_train,
        imputer,
        scaler,
        pca,
    ) = prepare_training_data(training)

    pair_models = build_pairwise_hulls(X_pc_train)

    first_file = resolve_monthly_file(START_YEAR, 1)
    global_ref = load_global_reference(first_file)

    (
        n_months,
        valid_month_count,
        supported_month_count,
        mean_extrapolation,
        supported_fraction_expected,
    ) = process_months(
        global_ref=global_ref,
        imputer=imputer,
        scaler=scaler,
        pca=pca,
        pair_models=pair_models,
    )

    result = global_ref.copy()

    result["N_months_expected"] = n_months
    result["N_valid_months_PCA_extrapolation"] = valid_month_count
    result["N_months_PCA_supported"] = supported_month_count

    result["PCA_extrapolation_mean"] = mean_extrapolation
    result["PCA_coverage_mean"] = 1.0 - mean_extrapolation

    result["PCA_supported_month_fraction_expected"] = (
        supported_fraction_expected
    )

    result["Retain_support_ge_50pct_expected_months"] = (
        supported_fraction_expected >= MIN_SUPPORTED_MONTH_FRACTION
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
        float_format="%.10g",
    )

    retained = result[
        result["Retain_support_ge_50pct_expected_months"]
    ]

    print("\n" + "=" * 100)
    print("Finished")
    print("=" * 100)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Global lakes: {len(result):,}")
    print(f"PCs retained: {pca.n_components_}")
    print(f"Valid PC pairs: {len(pair_models)}")
    print(
        "Retained lakes (>=50% supported months): "
        f"{len(retained):,}/{len(result):,} "
        f"({len(retained) / len(result) * 100:.2f}%)"
    )


if __name__ == "__main__":
    main()