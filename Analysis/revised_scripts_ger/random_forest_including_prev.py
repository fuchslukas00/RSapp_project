from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

# ============================================================
# RANDOM FOREST WITH LEAVE-ONE-YEAR-OUT CROSS-VALIDATION
# TWO MODEL VARIANTS:
#   1) without previous-year yield
#   2) with previous-year yield (yield_prev)
#
# Important:
# - Missing feature values are dropped per model variant
# - yield_prev is assumed to already exist in the merged CSV
# - Because yield_prev was built from 2016-2023 in the merge step,
#   both variants can use 2017-2023
# ============================================================


# ------------------------------------------------------------
# 1) FILE PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(
    r"C:\Users\lukas\Documents\Studium\Remote_Sensing_Products\project\RSapp_project"
)
DATA_CSV = PROJECT_DIR / "Analysis" / "merged_csv_ger" / "model_table_ww_2017_2023_ger.csv"

OUTPUT_DIR = PROJECT_DIR / "Analysis" / "model_results_ger"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2) SETTINGS
# ------------------------------------------------------------

DISTRICT_ID_COL = "nuts_id"
YEAR_COL = "year"
TARGET_COL = "yield_value"
PREV_YIELD_COL = "yield_prev"

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}


# ------------------------------------------------------------
# 3) HELPER FUNCTIONS
# ------------------------------------------------------------

def load_and_prepare_data(data_csv: Path) -> pd.DataFrame:
    """Load modelling table and perform basic cleaning."""
    df = pd.read_csv(data_csv)

    required_cols = {DISTRICT_ID_COL, YEAR_COL, TARGET_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[DISTRICT_ID_COL] = df[DISTRICT_ID_COL].astype(str).str.strip()
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    if PREV_YIELD_COL in df.columns:
        df[PREV_YIELD_COL] = pd.to_numeric(df[PREV_YIELD_COL], errors="coerce")

    df = df.dropna(subset=[DISTRICT_ID_COL, YEAR_COL, TARGET_COL]).copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)

    return df


def get_feature_columns(df: pd.DataFrame, include_prev_yield: bool) -> list:
    """
    Build feature list.
    If include_prev_yield=False, exclude yield_prev explicitly.
    """
    exclude_cols = {DISTRICT_ID_COL, YEAR_COL, TARGET_COL}

    if not include_prev_yield:
        exclude_cols.add(PREV_YIELD_COL)

    candidate_cols = [c for c in df.columns if c not in exclude_cols]

    numeric_feature_cols = []
    for col in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_feature_cols.append(col)

    return numeric_feature_cols


def run_leave_one_year_out_rf(
    df: pd.DataFrame,
    feature_cols: list,
    run_name: str,
    output_dir: Path,
):
    """
    Train and evaluate Random Forest with leave-one-year-out CV.
    Saves:
      - metrics
      - predictions
      - RF built-in feature importance
      - permutation importance
    """
    if len(feature_cols) == 0:
        raise ValueError(f"[{run_name}] No numeric features available for training.")

    n_missing = df[feature_cols].isna().sum().sum()
    if n_missing > 0:
        raise ValueError(
            f"[{run_name}] Feature matrix still contains {n_missing} missing values."
        )

    years = sorted(df[YEAR_COL].unique())

    print("\n" + "=" * 70)
    print(f"RUN: {run_name}")
    print("=" * 70)
    print(f"Rows: {len(df)}")
    print(f"Years: {years}")
    print(f"District ID column: {DISTRICT_ID_COL}")
    print(f"Target column: {TARGET_COL}")
    print(f"Number of features: {len(feature_cols)}")
    print("Features:")
    print(feature_cols)

    metrics_rows = []
    prediction_rows = []
    rf_importance_rows = []
    perm_importance_rows = []

    for test_year in years:
        print(f"\n=== Fold: test year = {test_year} | run = {run_name} ===")

        train_mask = df[YEAR_COL] != test_year
        test_mask = df[YEAR_COL] == test_year

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, TARGET_COL]

        X_test = df.loc[test_mask, feature_cols]
        y_test = df.loc[test_mask, TARGET_COL]

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Skipping year {test_year} because train or test set is empty.")
            continue

        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        metrics_rows.append({
            "run_name": run_name,
            "test_year": test_year,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae,
        })

        fold_predictions = pd.DataFrame({
            "run_name": run_name,
            DISTRICT_ID_COL: df.loc[test_mask, DISTRICT_ID_COL].values,
            YEAR_COL: df.loc[test_mask, YEAR_COL].values,
            "y_true": y_test.values,
            "y_pred": y_pred,
            "fold_test_year": test_year,
        })

        if PREV_YIELD_COL in df.columns:
            fold_predictions[PREV_YIELD_COL] = df.loc[test_mask, PREV_YIELD_COL].values

        prediction_rows.append(fold_predictions)

        rf_fold_importance = pd.DataFrame({
            "run_name": run_name,
            "feature": feature_cols,
            "importance": model.feature_importances_,
            "test_year": test_year,
        })
        rf_importance_rows.append(rf_fold_importance)

        perm = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=20,
            random_state=42,
            n_jobs=-1,
            scoring="r2",
        )

        perm_fold_importance = pd.DataFrame({
            "run_name": run_name,
            "feature": feature_cols,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
            "test_year": test_year,
        })
        perm_importance_rows.append(perm_fold_importance)

    if len(metrics_rows) == 0:
        raise ValueError(f"[{run_name}] No CV results were created.")

    metrics_df = pd.DataFrame(metrics_rows).sort_values("test_year").reset_index(drop=True)

    summary_row = pd.DataFrame([{
        "run_name": run_name,
        "test_year": "mean",
        "n_train": metrics_df["n_train"].mean(),
        "n_test": metrics_df["n_test"].mean(),
        "R2": metrics_df["R2"].mean(),
        "RMSE": metrics_df["RMSE"].mean(),
        "MAE": metrics_df["MAE"].mean(),
    }])

    metrics_out = pd.concat([metrics_df, summary_row], ignore_index=True)

    print("\n=== Yearly performance ===")
    print(metrics_out)

    rf_importance_df = pd.concat(rf_importance_rows, ignore_index=True)
    rf_importance_summary = (
        rf_importance_df
        .groupby(["run_name", "feature"], as_index=False)
        .agg(
            importance_mean=("importance", "mean"),
            importance_std=("importance", "std"),
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    perm_importance_df = pd.concat(perm_importance_rows, ignore_index=True)
    perm_importance_summary = (
        perm_importance_df
        .groupby(["run_name", "feature"], as_index=False)
        .agg(
            importance_mean=("importance_mean", "mean"),
            importance_std=("importance_mean", "std"),
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    print("\n=== Top RF feature importance ===")
    print(rf_importance_summary.head(15))

    print("\n=== Top permutation importance ===")
    print(perm_importance_summary.head(15))

    predictions_df = pd.concat(prediction_rows, ignore_index=True)

    metrics_csv = output_dir / f"rf_leave_one_year_out_metrics_{run_name}.csv"
    rf_importance_csv = output_dir / f"rf_feature_importance_mean_{run_name}.csv"
    perm_importance_csv = output_dir / f"rf_permutation_importance_mean_{run_name}.csv"
    predictions_csv = output_dir / f"rf_leave_one_year_out_predictions_{run_name}.csv"

    metrics_out.to_csv(metrics_csv, index=False)
    rf_importance_summary.to_csv(rf_importance_csv, index=False)
    perm_importance_summary.to_csv(perm_importance_csv, index=False)
    predictions_df.to_csv(predictions_csv, index=False)

    print("\nSaved files:")
    print(metrics_csv)
    print(rf_importance_csv)
    print(perm_importance_csv)
    print(predictions_csv)

    return {
        "metrics": metrics_out,
        "rf_importance": rf_importance_summary,
        "perm_importance": perm_importance_summary,
        "predictions": predictions_df,
    }


# ------------------------------------------------------------
# 4) MAIN
# ------------------------------------------------------------

def main():
    df = load_and_prepare_data(DATA_CSV)

    print("\nInitial table shape:")
    print(df.shape)

    print("\nMissing values per column before model-specific dropping:")
    print(df.isna().sum()[df.isna().sum() > 0].sort_values(ascending=False))

    # --------------------------------------------------------
    # RUN 1: WITHOUT previous-year yield
    # --------------------------------------------------------
    df_no_prev = df.copy()
    feature_cols_no_prev = get_feature_columns(df_no_prev, include_prev_yield=False)

    rows_before = len(df_no_prev)
    df_no_prev = df_no_prev.dropna(subset=feature_cols_no_prev).copy()
    rows_after = len(df_no_prev)

    print(f"\nDropped {rows_before - rows_after} rows due to missing feature values (no_prev run).")

    results_no_prev = run_leave_one_year_out_rf(
        df=df_no_prev,
        feature_cols=feature_cols_no_prev,
        run_name="ger_no_prev_yield",
        output_dir=OUTPUT_DIR,
    )

    # --------------------------------------------------------
    # RUN 2: WITH previous-year yield
    # --------------------------------------------------------
    df_with_prev = df.copy()
    feature_cols_with_prev = get_feature_columns(df_with_prev, include_prev_yield=True)

    rows_before = len(df_with_prev)
    df_with_prev = df_with_prev.dropna(subset=feature_cols_with_prev).copy()
    rows_after = len(df_with_prev)

    print(f"\nDropped {rows_before - rows_after} rows due to missing feature values (with_prev run).")

    results_with_prev = run_leave_one_year_out_rf(
        df=df_with_prev,
        feature_cols=feature_cols_with_prev,
        run_name="ger_with_prev_yield",
        output_dir=OUTPUT_DIR,
    )

    mean_r2_no_prev = results_no_prev["metrics"].loc[
        results_no_prev["metrics"]["test_year"] == "mean", "R2"
    ].values[0]

    mean_r2_with_prev = results_with_prev["metrics"].loc[
        results_with_prev["metrics"]["test_year"] == "mean", "R2"
    ].values[0]

    print("\n" + "=" * 70)
    print("COMPARISON OF BOTH RUNS")
    print("=" * 70)
    print(f"Mean R² without previous-year yield: {mean_r2_no_prev:.4f}")
    print(f"Mean R² with previous-year yield:    {mean_r2_with_prev:.4f}")


if __name__ == "__main__":
    main()