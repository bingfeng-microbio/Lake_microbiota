# -*- coding: utf-8 -*-
"""
Merged machine-learning script

This script combines the two uploaded scripts:
1) Multi-model comparison: XGBoost, Random Forest, Linear Regression, Lasso, KNN.
2) Random Forest RMSE calculation, model saving, and detailed CV-result export.

Recommended input format for your current lake microbiome dataset:
- One Excel file: data_for_ML.xlsx
- Predictor columns: FACTOR_COLS below
- Target columns: TARGET_COLS below

Main outputs:
- hyperparameter_tuning.xlsx
  * Model Results
  * All CV Results
  * Model Summary
  * Best Model Per Target
  * Statistical Tests
- models_merged/*.pkl
"""

import os
import re
import warnings

import numpy as np
import pandas as pd

from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# =============================================================================
# 1. Configuration
# =============================================================================

FILE_PATH = "data_for_ML.xlsx"
SHEET_NAME = 0  # first sheet. If needed, change to "Sheet1".

OUTPUT_EXCEL = "hyperparameter_tuning.xlsx"
MODEL_SAVE_PATH = "models"

# Use your explicitly defined lake predictors and target variables.
# This avoids accidental inclusion of SampleID, longitude, latitude, or target columns as predictors.
USE_EXPLICIT_COLUMNS = True

FACTOR_COLS = [
    "si10", "sp", "ssr", "t2m", "tp",
    "Lake_area", "Shore_dev", "Dis_avg", "Res_time", "Elevation",
    "Cropland", "Forest", "Grassland","Urban", "Barren"
]

TARGET_COLS = [
    "Shannon_arc","Shannon_bac","Shannon_fungi","Shannon_vir",
    "BC_arc","BC_bac","BC_fungi","BC_vir",
    "Carbon_fixation","Photosynthesis","ANR","Denitrification","DNR","Nitrification",
    "Nitrogen_fixation","ASR","DSR","Sulfur_oxidation",
]

# =============================================================================
# 2. Utility functions
# =============================================================================

def safe_filename(text: str) -> str:
    """Make a string safe for filenames."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def check_columns(data: pd.DataFrame, factor_cols, target_cols):
    """Check whether all required columns are present."""
    missing_factors = [c for c in factor_cols if c not in data.columns]
    missing_targets = [c for c in target_cols if c not in data.columns]

    if missing_factors:
        raise KeyError(
            "Missing predictor columns:\n"
            + "\n".join(missing_factors)
            + "\n\nPlease check FACTOR_COLS or set USE_EXPLICIT_COLUMNS = False."
        )

    if missing_targets:
        raise KeyError(
            "Missing target columns:\n"
            + "\n".join(missing_targets)
            + "\n\nPlease check TARGET_COLS or set USE_EXPLICIT_COLUMNS = False."
        )


def get_pipeline_step_name(model_name: str) -> str:
    """Match the pipeline step-name style used in the original scripts."""
    return model_name.lower().replace(" ", "")


def make_pipeline(model_name: str, model):
    """
    Build a model pipeline.
    The imputer handles missing predictor values.
    The scaler is retained from your original scripts.
    """
    step_name = get_pipeline_step_name(model_name)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (step_name, model)
    ])


def extract_best_cv_scores(grid_search: GridSearchCV) -> dict:
    """Extract split-wise CV scores of the best hyperparameter combination."""
    best_idx = grid_search.best_index_
    cv_results = grid_search.cv_results_

    split_scores = []
    for key in cv_results.keys():
        if key.startswith("split") and key.endswith("_test_score"):
            split_scores.append(float(cv_results[key][best_idx]))

    split_scores = np.array(split_scores, dtype=float)
    return {
        "CV_R2_mean_best": float(np.nanmean(split_scores)) if len(split_scores) > 0 else np.nan,
        "CV_R2_sd_best": float(np.nanstd(split_scores, ddof=1)) if len(split_scores) > 1 else np.nan,
        "CV_R2_scores_best": ";".join([f"{v:.6f}" for v in split_scores])
    }


def run_statistical_tests(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional statistical comparison among models using target-wise R2_test.

    This helps address reviewer concerns that model choice should not rely only on raw R2/RMSE.
    It compares models across target variables. If scipy is unavailable, the function returns
    an explanatory table instead of failing.
    """
    try:
        from scipy.stats import friedmanchisquare, wilcoxon
    except Exception:
        return pd.DataFrame([{
            "Test": "Not performed",
            "Comparison": "scipy unavailable",
            "Statistic": np.nan,
            "P_value": np.nan,
            "Note": "Install scipy to run Friedman and Wilcoxon signed-rank tests."
        }])

    if results_df.empty:
        return pd.DataFrame()

    pivot = results_df.pivot_table(
        index="Target Variable",
        columns="Model",
        values="R2_test",
        aggfunc="first"
    ).dropna(axis=0, how="any")

    test_rows = []

    if pivot.shape[0] >= 2 and pivot.shape[1] >= 3:
        model_names = list(pivot.columns)
        arrays = [pivot[m].values for m in model_names]
        stat, p = friedmanchisquare(*arrays)
        test_rows.append({
            "Test": "Friedman test",
            "Comparison": "All models based on target-wise R2_test",
            "Statistic": float(stat),
            "P_value": float(p),
            "Note": "Non-parametric paired comparison across target variables."
        })
    else:
        test_rows.append({
            "Test": "Friedman test",
            "Comparison": "All models based on target-wise R2_test",
            "Statistic": np.nan,
            "P_value": np.nan,
            "Note": "Not enough complete model-target results."
        })

    # Pairwise Random Forest vs each other model.
    if "Random Forest" in pivot.columns:
        p_values = []
        comparisons = []

        for other_model in pivot.columns:
            if other_model == "Random Forest":
                continue

            try:
                stat, p = wilcoxon(pivot["Random Forest"].values, pivot[other_model].values)
            except Exception:
                stat, p = np.nan, np.nan

            comparisons.append(other_model)
            p_values.append(p)

        # Benjamini-Hochberg FDR correction.
        p_values_array = np.array(p_values, dtype=float)
        valid = ~np.isnan(p_values_array)
        adjusted = np.full_like(p_values_array, np.nan, dtype=float)

        if valid.sum() > 0:
            valid_p = p_values_array[valid]
            order = np.argsort(valid_p)
            ranked_p = valid_p[order]
            m = len(ranked_p)
            bh = ranked_p * m / np.arange(1, m + 1)
            bh = np.minimum.accumulate(bh[::-1])[::-1]
            bh = np.minimum(bh, 1.0)
            adjusted_valid = np.empty_like(bh)
            adjusted_valid[order] = bh
            adjusted[valid] = adjusted_valid

        for other_model, p, p_adj in zip(comparisons, p_values, adjusted):
            try:
                stat, raw_p = wilcoxon(pivot["Random Forest"].values, pivot[other_model].values)
            except Exception:
                stat, raw_p = np.nan, np.nan

            test_rows.append({
                "Test": "Wilcoxon signed-rank test",
                "Comparison": f"Random Forest vs {other_model}",
                "Statistic": float(stat) if not np.isnan(stat) else np.nan,
                "P_value": float(raw_p) if not np.isnan(raw_p) else np.nan,
                "P_value_BH_FDR": float(p_adj) if not np.isnan(p_adj) else np.nan,
                "Note": "Paired comparison across target variables using R2_test."
            })

    return pd.DataFrame(test_rows)


# =============================================================================
# 3. Main workflow
# =============================================================================

def main():
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    print("=" * 100)
    print("Loading data")
    print("=" * 100)
    print(f"Input file: {FILE_PATH}")

    data = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
    print(f"Data shape: {data.shape}")

    if USE_EXPLICIT_COLUMNS:
        check_columns(data, FACTOR_COLS, TARGET_COLS)
        X_all = data[FACTOR_COLS].copy()
        y_columns = TARGET_COLS
    else:
        X_all = data.iloc[:, FALLBACK_X_START:FALLBACK_X_END].copy()
        y_columns = list(data.columns[FALLBACK_Y_START:])

    # Convert predictors to numeric. Non-numeric values become NaN and are imputed in the pipeline.
    X_all = X_all.apply(pd.to_numeric, errors="coerce")

    print(f"Predictor columns ({len(X_all.columns)}): {list(X_all.columns)}")
    print(f"Target columns ({len(y_columns)}): {list(y_columns)}")

    TEST_SIZE = 0.10
    RANDOM_STATE = 43
    CV_FOLDS = 10
    N_JOBS = -1

    # -------------------------------------------------------------------------
    # Models and hyperparameter grids: merged from both uploaded scripts.
    # -------------------------------------------------------------------------
    models = {
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0
        ),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "Linear Regression": LinearRegression(),
        "Lasso": Lasso(random_state=RANDOM_STATE),
        "K Nearest Neighbors": KNeighborsRegressor()
    }

    param_grids = {
        "XGBoost": {
            "xgboost__n_estimators": [100, 200, 300, 500, 600],
            "xgboost__max_depth": [3, 5, 7, 9],
            "xgboost__learning_rate": [0.01, 0.1, 0.2, 0.3]
        },
        "Random Forest": {
            "randomforest__n_estimators": [100, 200, 300, 500, 600],
            "randomforest__max_depth": [None, 10, 20, 30, 40],
            "randomforest__min_samples_split": [2, 5]
        },
        "Linear Regression": {},
        "Lasso": {
            "lasso__alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
            "lasso__max_iter": [5000, 10000, 15000, 30000]
        },
        "K Nearest Neighbors": {
            "knearestneighbors__n_neighbors": [3, 5, 7, 10],
            "knearestneighbors__weights": ["uniform", "distance"],
            "knearestneighbors__metric": ["euclidean", "manhattan"]
        }
    }

    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results = []
    all_cv_results = []

    # -------------------------------------------------------------------------
    # Model training
    # -------------------------------------------------------------------------
    for y_column in y_columns:
        print("\n" + "=" * 100)
        print(f"Target variable: {y_column}")
        print("=" * 100)

        y_all = pd.to_numeric(data[y_column], errors="coerce")

        # Drop samples with missing target values. Predictor missing values are handled by SimpleImputer.
        valid_mask = y_all.notna()
        X = X_all.loc[valid_mask].copy()
        y = y_all.loc[valid_mask].copy()

        if len(y) < CV_FOLDS + 5:
            print(f"Skipping {y_column}: too few valid samples ({len(y)}).")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        for model_name, model in models.items():
            print(f"\nRunning model: {model_name}")

            pipeline = make_pipeline(model_name, model)

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grids[model_name],
                cv=cv,
                scoring="r2",
                refit=True,
                n_jobs=N_JOBS,
                return_train_score=True
            )

            grid_search.fit(X_train, y_train)

            # Full CV results for all hyperparameter combinations.
            cv_results = pd.DataFrame(grid_search.cv_results_)
            cv_results["Model"] = model_name
            cv_results["Target Variable"] = y_column
            all_cv_results.append(cv_results)

            best_params = grid_search.best_params_
            best_cv = extract_best_cv_scores(grid_search)

            # Predictions.
            y_pred_train = grid_search.predict(X_train)
            y_pred_test = grid_search.predict(X_test)
            y_pred_overall = grid_search.predict(X)

            # Metrics.
            r2_train = r2_score(y_train, y_pred_train)
            rmse_train = rmse(y_train, y_pred_train)
            mae_train = mean_absolute_error(y_train, y_pred_train)

            r2_test = r2_score(y_test, y_pred_test)
            rmse_test = rmse(y_test, y_pred_test)
            mae_test = mean_absolute_error(y_test, y_pred_test)

            r2_overall = r2_score(y, y_pred_overall)
            rmse_overall = rmse(y, y_pred_overall)
            mae_overall = mean_absolute_error(y, y_pred_overall)

            overfit_gap = r2_train - r2_test

            # Save best model.
            model_filename = os.path.join(
                MODEL_SAVE_PATH,
                f"{safe_filename(y_column)}_{safe_filename(model_name)}_best_model.pkl"
            )
            dump(grid_search.best_estimator_, model_filename)

            # Store model-level results.
            results.append({
                "Target Variable": y_column,
                "Model": model_name,
                "Best Parameters": str(best_params),
                "CV_R2_mean_best": best_cv["CV_R2_mean_best"],
                "CV_R2_sd_best": best_cv["CV_R2_sd_best"],
                "CV_R2_scores_best": best_cv["CV_R2_scores_best"],
                "R2_train": r2_train,
                "RMSE_train": rmse_train,
                "MAE_train": mae_train,
                "R2_test": r2_test,
                "RMSE_test": rmse_test,
                "MAE_test": mae_test,
                "R2_overall": r2_overall,
                "RMSE_overall": rmse_overall,
                "MAE_overall": mae_overall,
                "Overfit_gap_train_minus_test_R2": overfit_gap,
                "Saved Model": model_filename,
                "N_samples": len(y),
                "N_train": len(y_train),
                "N_test": len(y_test)
            })

            print(f"Best Parameters: {best_params}")
            print(f"CV R2 best: {best_cv['CV_R2_mean_best']:.4f} ± {best_cv['CV_R2_sd_best']:.4f}")
            print(f"Train:   R2={r2_train:.4f}, RMSE={rmse_train:.4f}, MAE={mae_train:.4f}")
            print(f"Test:    R2={r2_test:.4f}, RMSE={rmse_test:.4f}, MAE={mae_test:.4f}")
            print(f"Overall: R2={r2_overall:.4f}, RMSE={rmse_overall:.4f}, MAE={mae_overall:.4f}")
            print(f"Model saved: {model_filename}")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    all_cv_results_df = pd.concat(all_cv_results, ignore_index=True) if all_cv_results else pd.DataFrame()

    if not results_df.empty:
        model_summary_df = (
            results_df.groupby("Model")
            .agg(
                Mean_R2_test=("R2_test", "mean"),
                SD_R2_test=("R2_test", "std"),
                Median_R2_test=("R2_test", "median"),
                Mean_RMSE_test=("RMSE_test", "mean"),
                Mean_MAE_test=("MAE_test", "mean"),
                Mean_CV_R2=("CV_R2_mean_best", "mean"),
                Number_of_targets=("Target Variable", "count")
            )
            .reset_index()
            .sort_values("Mean_R2_test", ascending=False)
        )

        # Best model per target according to external test R2.
        best_model_per_target_df = (
            results_df.sort_values(["Target Variable", "R2_test"], ascending=[True, False])
            .groupby("Target Variable", as_index=False)
            .head(1)
            .sort_values("R2_test", ascending=False)
        )

        statistical_tests_df = run_statistical_tests(results_df)
    else:
        model_summary_df = pd.DataFrame()
        best_model_per_target_df = pd.DataFrame()
        statistical_tests_df = pd.DataFrame()

    with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
        results_df.to_excel(writer, sheet_name="Model Results", index=False)
        all_cv_results_df.to_excel(writer, sheet_name="All CV Results", index=False)
        model_summary_df.to_excel(writer, sheet_name="Model Summary", index=False)
        best_model_per_target_df.to_excel(writer, sheet_name="Best Model Per Target", index=False)
        statistical_tests_df.to_excel(writer, sheet_name="Statistical Tests", index=False)

    print("\n" + "=" * 100)
    print("Finished")
    print("=" * 100)
    print(f"Results saved to: {OUTPUT_EXCEL}")
    print(f"Models saved to: {MODEL_SAVE_PATH}/")

    if not model_summary_df.empty:
        print("\nModel summary based on test R2:")
        print(model_summary_df[["Model", "Mean_R2_test", "SD_R2_test", "Mean_RMSE_test", "Number_of_targets"]])


if __name__ == "__main__":
    main()