from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

# ============================================================
# RANDOM FOREST WITH LEAVE-ONE-YEAR-OUT CROSS-VALIDATION
# BASELINE MODEL: PREVIOUS-YEAR YIELD ONLY
#
# This script:
# 1. Loads the merged modelling table
# 2. Uses only yield_prev as predictor
# 3. Trains a Random Forest in leave-one-year-out CV
# 4. Saves:
#    - yearly metrics
#    - predictions
#    - RF built-in feature importance
#    - permutation importance
# ============================================================


# ------------------------------------------------------------
# 1) FILE PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(
    r"C:\Users\lukas\Documents\Studium\Remote_Sensing_Products\project\RSapp_project"
)
DATA_CSV = PROJECT_DIR / "Analysis" / "merged_csv_ger" / "model_table_ww_2017_2023_ger.csv"

OUTPUT_DIR = PROJECT_DIR / "Analysis" / "model_results_ger_prev_only"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2) SETTINGS
# ------------------------------------------------------------

DISTRICT_ID_COL = "nuts_id"
YEAR_COL = "year"
TARGET_COL = "yield_value"
FEATURE_COL = "yield_prev"

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": 1,   # only one feature
    "random_state": 42,
    "n_jobs": -1,
}


# ------------------------------------------------------------
# 3) HELPER FUNCTIONS
# ------------------------------------------------------------

def load_and_prepare_data(data_csv: Path) -> pd.DataFrame:
    """Load modelling table and perform basic cleaning."""
    df = pd.read_csv(data_csv)

    required_cols = {DISTRICT_ID_COL, YEAR_COL, TARGET_COL, FEATURE_COL}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[DISTRICT_ID_COL] = df[DISTRICT_ID_COL].astype(str).str.strip()
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df[FEATURE_COL] = pd.to_numeric(df[FEATURE_COL], errors="coerce")

    df = df.dropna(subset=[DISTRICT_ID_COL, YEAR_COL, TARGET_COL, FEATURE_COL]).copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)

    return df


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
            FEATURE_COL: df.loc[test_mask, FEATURE_COL].values,
        })
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

    print("\n=== RF feature importance ===")
    print(rf_importance_summary)

    print("\n=== Permutation importance ===")
    print(perm_importance_summary)

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

    print("\nMissing values per column after loading:")
    print(df.isna().sum()[df.isna().sum() > 0].sort_values(ascending=False))

    feature_cols = [FEATURE_COL]

    results = run_leave_one_year_out_rf(
        df=df,
        feature_cols=feature_cols,
        run_name="ger_prev_yield_only",
        output_dir=OUTPUT_DIR,
    )

    mean_r2 = results["metrics"].loc[
        results["metrics"]["test_year"] == "mean", "R2"
    ].values[0]

    print("\n" + "=" * 70)
    print("PREVIOUS-YEAR-YIELD-ONLY BASELINE")
    print("=" * 70)
    print(f"Mean R²: {mean_r2:.4f}")


if __name__ == "__main__":
    main()