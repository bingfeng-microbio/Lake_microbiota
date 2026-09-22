# -*- coding: utf-8 -*-
"""
Predict global lake microbial attributes for every month from 2001-01 to 2020-12,
then calculate a 20-year historical baseline for each lake.

Workflow
--------
For each saved Random Forest model:
1. Load the model once.
2. Read the 240 monthly predictor tables in chronological order.
3. Predict every lake for every month.
4. Calculate lake-level historical mean, SD, minimum, maximum and valid-month count.
5. Merge the 20-year mean predictions of all targets into one final table.

Important
---------
The historical baseline is calculated as:

    mean[f(X_month)]

not:

    f(mean[X_month])

This distinction matters because Random Forest is nonlinear.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd


# =============================================================================
# 1. Configuration
# =============================================================================

BASE_DIR = Path("/Users/bingfengchen/Desktop/Freshwater")

MODEL_DIR = BASE_DIR / "models"
MODEL_LIST_FILE = BASE_DIR / "model_paths.txt"
PREDICTOR_DIR = BASE_DIR / "global_lake_factors_2001_2020_by_month_converted"

OUTPUT_DIR = BASE_DIR / "predictions_2001_2020"
SUMMARY_OUTPUT_DIR = OUTPUT_DIR / "historical_summary_by_target"

START_YEAR = 2001
END_YEAR = 2020

PREDICTOR_FILE_TEMPLATE = "global_lake_factors_{year}-{month:02d}.csv"

# Require all monthly predictor files for the specified period.
REQUIRE_COMPLETE_240_MONTHS = True

# If a model cannot predict rows containing missing features,
# predict complete rows only and leave incomplete rows as NA.
ALLOW_INCOMPLETE_ROWS_AS_NA = True

FLOAT_FORMAT = "%.10g"

ID_COL = "Hylak_id"
COORD_COLS = ["Longitude", "Latitude"]


# =============================================================================
# 2. General utilities
# =============================================================================

def require_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    object_name: str,
) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(
            f"{object_name} is missing required columns:\n"
            + "\n".join(f"  - {col}" for col in missing)
            + f"\n\nAvailable columns:\n{list(df.columns)}"
        )


def normalize_hylak_id(df: pd.DataFrame, object_name: str) -> pd.DataFrame:
    out = df.copy()
    values = pd.to_numeric(out[ID_COL], errors="coerce")

    if values.isna().any():
        raise ValueError(
            f"{object_name}: {int(values.isna().sum())} rows contain "
            f"missing or non-numeric {ID_COL}."
        )

    rounded = np.round(values.to_numpy(dtype=float))
    if not np.allclose(values.to_numpy(dtype=float), rounded):
        raise ValueError(f"{object_name}: {ID_COL} contains non-integer values.")

    out[ID_COL] = rounded.astype("int64")
    return out


def resolve_predictor_file(year: int, month: int) -> Path:
    plain = PREDICTOR_DIR / PREDICTOR_FILE_TEMPLATE.format(
        year=year,
        month=month,
    )
    gzip_path = plain.with_suffix(plain.suffix + ".gz")

    if plain.exists():
        return plain
    if gzip_path.exists():
        return gzip_path

    raise FileNotFoundError(
        f"Monthly predictor file was not found:\n"
        f"{plain}\n"
        f"or\n"
        f"{gzip_path}"
    )


def list_monthly_files() -> list[tuple[int, int, Path]]:
    files: list[tuple[int, int, Path]] = []
    missing: list[str] = []

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            try:
                files.append((year, month, resolve_predictor_file(year, month)))
            except FileNotFoundError:
                expected = PREDICTOR_DIR / PREDICTOR_FILE_TEMPLATE.format(
                    year=year,
                    month=month,
                )
                missing.append(str(expected))

    expected_n = (END_YEAR - START_YEAR + 1) * 12

    if missing and REQUIRE_COMPLETE_240_MONTHS:
        preview = "\n".join(f"  - {x}" for x in missing[:20])
        more = (
            f"\n  ... and {len(missing) - 20} more"
            if len(missing) > 20
            else ""
        )
        raise FileNotFoundError(
            f"{len(missing)} monthly predictor files are missing:\n"
            f"{preview}{more}"
        )

    if not files:
        raise FileNotFoundError(
            f"No monthly predictor files were found in:\n{PREDICTOR_DIR}"
        )

    print(
        f"Monthly predictor files found: {len(files)}/{expected_n} "
        f"({START_YEAR}-01 to {END_YEAR}-12)"
    )
    return files


# =============================================================================
# 3. Model utilities
# =============================================================================

def read_target_identifiers() -> list[str]:
    """
    Prefer model_paths.txt. If absent, infer identifiers from filenames:
    *_Random_Forest_best_model.pkl
    """
    if MODEL_LIST_FILE.exists():
        identifiers = [
            line.strip()
            for line in MODEL_LIST_FILE.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    else:
        suffix = "_Random_Forest_best_model.pkl"
        identifiers = sorted(
            path.name[:-len(suffix)]
            for path in MODEL_DIR.glob(f"*{suffix}")
        )
        print(
            f"WARNING: {MODEL_LIST_FILE} was not found. "
            f"Detected {len(identifiers)} models from filenames."
        )

    if not identifiers:
        raise ValueError(
            "No target identifiers were found in model_paths.txt "
            "and no matching model files were detected."
        )

    duplicates = pd.Series(identifiers).duplicated()
    if duplicates.any():
        duplicated_names = pd.Series(identifiers)[duplicates].tolist()
        raise ValueError(
            f"model_paths.txt contains duplicated identifiers: {duplicated_names}"
        )

    return identifiers


def get_model_feature_names(model) -> list[str]:
    """Extract training feature names from a fitted estimator or Pipeline."""
    if hasattr(model, "feature_names_in_"):
        return [str(x) for x in model.feature_names_in_]

    if hasattr(model, "named_steps"):
        for _, step in reversed(list(model.named_steps.items())):
            if hasattr(step, "feature_names_in_"):
                return [str(x) for x in step.feature_names_in_]

    raise AttributeError(
        "The loaded model does not contain feature_names_in_. "
        "The model should be fitted with a pandas DataFrame so that the exact "
        "training feature names and order are preserved."
    )


def model_has_imputer(model) -> bool:
    if hasattr(model, "named_steps"):
        return any(
            "imput" in str(name).lower()
            for name in model.named_steps.keys()
        )
    return False


# =============================================================================
# 4. Prediction and summary utilities
# =============================================================================

def predict_safely(
    model,
    X: pd.DataFrame,
    object_name: str,
) -> np.ndarray:
    """
    Predict all rows when possible. If the estimator cannot handle missing values,
    predict complete rows only and return NA for incomplete rows.
    """
    X_numeric = X.apply(pd.to_numeric, errors="coerce")
    output = np.full(len(X_numeric), np.nan, dtype=np.float64)

    if model_has_imputer(model):
        output[:] = np.asarray(model.predict(X_numeric), dtype=float)
        return output

    complete_mask = X_numeric.notna().all(axis=1).to_numpy()

    if complete_mask.all():
        output[:] = np.asarray(model.predict(X_numeric), dtype=float)
        return output

    if not ALLOW_INCOMPLETE_ROWS_AS_NA:
        missing_by_feature = X_numeric.isna().sum()
        missing_by_feature = missing_by_feature[missing_by_feature > 0]
        raise ValueError(
            f"{object_name} contains missing predictor values:\n"
            f"{missing_by_feature.to_string()}"
        )

    n_incomplete = int((~complete_mask).sum())
    print(
        f"      WARNING: {n_incomplete:,} rows have incomplete predictors "
        "and will receive NA predictions."
    )

    if complete_mask.any():
        output[complete_mask] = np.asarray(
            model.predict(X_numeric.loc[complete_mask]),
            dtype=float,
        )

    return output


def update_accumulators(
    prediction: np.ndarray,
    count: np.ndarray,
    value_sum: np.ndarray,
    value_sumsq: np.ndarray,
    value_min: np.ndarray,
    value_max: np.ndarray,
) -> None:
    valid = np.isfinite(prediction)
    if not valid.any():
        return

    values = prediction[valid]

    count[valid] += 1
    value_sum[valid] += values
    value_sumsq[valid] += values * values

    current_min = value_min[valid]
    current_max = value_max[valid]

    value_min[valid] = np.where(
        np.isnan(current_min),
        values,
        np.minimum(current_min, values),
    )
    value_max[valid] = np.where(
        np.isnan(current_max),
        values,
        np.maximum(current_max, values),
    )


def finalize_summary(
    lake_reference: pd.DataFrame,
    identifier: str,
    count: np.ndarray,
    value_sum: np.ndarray,
    value_sumsq: np.ndarray,
    value_min: np.ndarray,
    value_max: np.ndarray,
) -> pd.DataFrame:
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.divide(
            value_sum,
            count,
            out=np.full_like(value_sum, np.nan, dtype=float),
            where=count > 0,
        )

        variance_numerator = value_sumsq - (
            value_sum * value_sum / np.maximum(count, 1)
        )
        variance = np.divide(
            variance_numerator,
            count - 1,
            out=np.full_like(value_sum, np.nan, dtype=float),
            where=count > 1,
        )
        variance = np.maximum(variance, 0.0)
        sd = np.sqrt(variance)

    summary = lake_reference.copy()
    summary[f"{identifier}_historical_mean"] = mean
    summary[f"{identifier}_historical_sd"] = sd
    summary[f"{identifier}_historical_min"] = value_min
    summary[f"{identifier}_historical_max"] = value_max
    summary[f"{identifier}_valid_months"] = count

    return summary


# =============================================================================
# 5. Main prediction workflow
# =============================================================================

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    monthly_files = list_monthly_files()
    identifiers = read_target_identifiers()

    print("=" * 100)
    print("Historical monthly prediction: 2001-2020")
    print("=" * 100)
    print(f"Model directory          : {MODEL_DIR}")
    print(f"Monthly predictor folder : {PREDICTOR_DIR}")
    print(f"Number of targets        : {len(identifiers)}")
    print()

    # Use the first monthly file to define the authoritative lake order and coordinates.
    _, _, first_path = monthly_files[0]
    first_data = pd.read_csv(
        first_path,
        usecols=[ID_COL] + COORD_COLS,
        low_memory=False,
    )
    require_columns(
        first_data,
        [ID_COL] + COORD_COLS,
        f"Predictor file {first_path.name}",
    )
    first_data = normalize_hylak_id(first_data, first_path.name)

    if first_data[ID_COL].duplicated().any():
        raise ValueError(
            f"{first_path.name} contains duplicated {ID_COL} values."
        )

    first_data = first_data.reset_index(drop=True)

    master_ids = first_data[ID_COL].to_numpy(dtype=np.int64)
    master_index = pd.Index(master_ids, name=ID_COL)
    lake_reference = first_data[[ID_COL] + COORD_COLS].copy()

    n_lakes = len(lake_reference)
    n_expected_months = len(monthly_files)

    combined_means = lake_reference.copy()
    run_summary_records: list[dict] = []

    for target_number, identifier in enumerate(identifiers, start=1):
        print("\n" + "=" * 100)
        print(f"Target {target_number}/{len(identifiers)}: {identifier}")
        print("=" * 100)

        model_path = MODEL_DIR / f"{identifier}_Random_Forest_best_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file was not found:\n{model_path}"
            )

        print(f"Loading model: {model_path.name}")
        model = joblib.load(model_path)

        features = get_model_feature_names(model)
        print(f"Training feature count: {len(features)}")
        print(f"Training feature order: {features}")

        count = np.zeros(n_lakes, dtype=np.int16)
        value_sum = np.zeros(n_lakes, dtype=np.float64)
        value_sumsq = np.zeros(n_lakes, dtype=np.float64)
        value_min = np.full(n_lakes, np.nan, dtype=np.float64)
        value_max = np.full(n_lakes, np.nan, dtype=np.float64)

        for month_number, (year, month, input_path) in enumerate(
            monthly_files,
            start=1,
        ):
            print(
                f"  [{month_number:03d}/{n_expected_months}] "
                f"{year}-{month:02d}"
            )

            monthly = pd.read_csv(
                input_path,
                usecols=lambda col: col in ({ID_COL} | set(features)),
                low_memory=False,
            )

            require_columns(
                monthly,
                [ID_COL] + features,
                f"Predictor file {input_path.name}",
            )
            monthly = normalize_hylak_id(monthly, input_path.name)

            if monthly[ID_COL].duplicated().any():
                raise ValueError(
                    f"{input_path.name} contains duplicated {ID_COL} values."
                )

            monthly = (
                monthly.set_index(ID_COL)
                .reindex(master_index)
                .reset_index()
            )

            prediction = predict_safely(
                model=model,
                X=monthly[features],
                object_name=f"{identifier}, {year}-{month:02d}",
            )

            update_accumulators(
                prediction=prediction,
                count=count,
                value_sum=value_sum,
                value_sumsq=value_sumsq,
                value_min=value_min,
                value_max=value_max,
            )

        summary = finalize_summary(
            lake_reference=lake_reference,
            identifier=identifier,
            count=count,
            value_sum=value_sum,
            value_sumsq=value_sumsq,
            value_min=value_min,
            value_max=value_max,
        )

        summary_file = (
            SUMMARY_OUTPUT_DIR
            / f"{identifier}_historical_2001_2020_summary.csv"
        )
        summary.to_csv(
            summary_file,
            index=False,
            encoding="utf-8-sig",
            float_format=FLOAT_FORMAT,
        )

        combined_means[identifier] = summary[
            f"{identifier}_historical_mean"
        ].to_numpy()

        run_summary_records.append({
            "Target": identifier,
            "Model_file": model_path.name,
            "N_features": len(features),
            "Features": "|".join(features),
            "N_global_lakes": n_lakes,
            "N_expected_months": n_expected_months,
            "N_lakes_with_all_months": int(
                (count == n_expected_months).sum()
            ),
            "N_lakes_with_any_prediction": int((count > 0).sum()),
            "Minimum_valid_months": int(count.min()),
            "Median_valid_months": float(np.median(count)),
            "Maximum_valid_months": int(count.max()),
        })

        pd.DataFrame(run_summary_records).to_csv(
            OUTPUT_DIR / "prediction_run_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )

        del model
        gc.collect()

        print(f"Saved target summary: {summary_file}")

    combined_mean_file = (
        OUTPUT_DIR
        / "historical_2001_2020_mean_predictions_all_targets.csv"
    )
    combined_means.to_csv(
        combined_mean_file,
        index=False,
        encoding="utf-8-sig",
        float_format=FLOAT_FORMAT,
    )

    print("\n" + "=" * 100)
    print("All historical predictions finished")
    print("=" * 100)
    print(f"Main 20-year baseline table:\n{combined_mean_file}")
    print(f"Target summaries:\n{SUMMARY_OUTPUT_DIR}")


if __name__ == "__main__":
    main()