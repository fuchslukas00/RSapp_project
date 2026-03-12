from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

# ============================================================
# RANDOM FOREST WITH LEAVE-ONE-YEAR-OUT CROSS-VALIDATION
# ============================================================
# This script:
# 1. Loads the final modelling table
# 2. Uses leave-one-year-out cross-validation
# 3. Trains one RandomForestRegressor per fold
# 4. Evaluates performance on the held-out year
# 5. Stores:
#    - yearly R²
#    - yearly RMSE
#    - yearly MAE
# 6. Computes feature importance
#    - built-in RF importance
#    - permutation importance on the test fold
# 7. Aggregates feature importance across folds
# 8. Saves results as CSV
# ============================================================


# ------------------------------------------------------------
# 1) FILE PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(r"C:\Users\lukas\Documents\Studium\Remote_Sensing_Products\project\RSapp_project")
DATA_CSV = PROJECT_DIR / "Analysis" / "merged_csv" / "model_table_ww_ger_2017_2023.csv"

OUTPUT_DIR = PROJECT_DIR / "Analysis" / "model_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS_CSV = OUTPUT_DIR / "rf_leave_one_year_out_metrics_ger.csv"
IMPORTANCE_RF_CSV = OUTPUT_DIR / "rf_feature_importance_mean_ger.csv"
IMPORTANCE_PERM_CSV = OUTPUT_DIR / "rf_permutation_importance_mean_ger.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "rf_leave_one_year_out_predictions_ger.csv"


# ------------------------------------------------------------
# 2) LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_CSV)

# Basic checks
DISTRICT_ID_COL = "NUTS_CODE"
TARGET_COL = "yield_value"

required_cols = {DISTRICT_ID_COL, "year", TARGET_COL}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df[DISTRICT_ID_COL] = df[DISTRICT_ID_COL].astype(str).str.strip()
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

# Drop rows without target
df = df.dropna(subset=[TARGET_COL]).copy()

# ------------------------------------------------------------
# 3) DEFINE FEATURES AND TARGET
# ------------------------------------------------------------
# Exclude identifier columns and target column.
# Everything else is treated as predictor.
# ------------------------------------------------------------

exclude_cols = [DISTRICT_ID_COL, "year", TARGET_COL]
feature_cols = [c for c in df.columns if c not in exclude_cols]

# Optional: if there are still non-numeric columns, remove them
numeric_feature_cols = []
for col in feature_cols:
    if pd.api.types.is_numeric_dtype(df[col]):
        numeric_feature_cols.append(col)

feature_cols = numeric_feature_cols

X_all = df[feature_cols].copy()
y_all = df[TARGET_COL].copy()

# Optional safety check: RF cannot handle NaN directly
n_missing = X_all.isna().sum().sum()
if n_missing > 0:
    raise ValueError(
        f"Feature matrix contains {n_missing} missing values. "
        "Handle missing values before training."
    )

print(f"Rows: {len(df)}")
print(f"Years: {sorted(df['year'].unique())}")
print(f"District ID column: {DISTRICT_ID_COL}")
print(f"Number of features: {len(feature_cols)}")
print("Features:")
print(feature_cols)


# ------------------------------------------------------------
# 4) RANDOM FOREST SETTINGS
# ------------------------------------------------------------
# Reasonable starting values for a medium-small dataset.
# ------------------------------------------------------------

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
# 5) LEAVE-ONE-YEAR-OUT CV
# ------------------------------------------------------------

years = sorted(df["year"].unique())

metrics_rows = []
prediction_rows = []

rf_importance_rows = []
perm_importance_rows = []

for test_year in years:
    print(f"\n=== Fold: test year = {test_year} ===")

    train_mask = df["year"] != test_year
    test_mask = df["year"] == test_year

    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, TARGET_COL]

    X_test = df.loc[test_mask, feature_cols]
    y_test = df.loc[test_mask, TARGET_COL]

    # Train model
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_train, y_train)

    # Predict on held-out year
    y_pred = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    metrics_rows.append({
        "test_year": test_year,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
    })

    # Store predictions
    fold_predictions = pd.DataFrame({
        DISTRICT_ID_COL: df.loc[test_mask, DISTRICT_ID_COL].values,
        "year": df.loc[test_mask, "year"].values,
        "y_true": y_test.values,
        "y_pred": y_pred,
        "fold_test_year": test_year,
    })
    prediction_rows.append(fold_predictions)

    # --------------------------------------------------------
    # 5.1 Built-in RF importance
    # --------------------------------------------------------
    rf_fold_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
        "test_year": test_year,
    })
    rf_importance_rows.append(rf_fold_importance)

    # --------------------------------------------------------
    # 5.2 Permutation importance on the held-out test year
    # --------------------------------------------------------
    # This is often more interpretable than built-in RF importance.
    # --------------------------------------------------------
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
        "feature": feature_cols,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
        "test_year": test_year,
    })
    perm_importance_rows.append(perm_fold_importance)


# ------------------------------------------------------------
# 6) BUILD METRICS TABLE
# ------------------------------------------------------------

metrics_df = pd.DataFrame(metrics_rows)
metrics_df = metrics_df.sort_values("test_year").reset_index(drop=True)

# Add summary row
summary_row = pd.DataFrame([{
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


# ------------------------------------------------------------
# 7) AGGREGATE FEATURE IMPORTANCE ACROSS FOLDS
# ------------------------------------------------------------

rf_importance_df = pd.concat(rf_importance_rows, ignore_index=True)
rf_importance_summary = (
    rf_importance_df
    .groupby("feature", as_index=False)
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
    .groupby("feature", as_index=False)
    .agg(
        importance_mean=("importance_mean", "mean"),
        importance_std=("importance_mean", "std"),
    )
    .sort_values("importance_mean", ascending=False)
    .reset_index(drop=True)
)

print("\n=== Mean RF feature importance across folds ===")
print(rf_importance_summary.head(15))

print("\n=== Mean permutation importance across folds ===")
print(perm_importance_summary.head(15))


# ------------------------------------------------------------
# 8) SAVE OUTPUTS
# ------------------------------------------------------------

predictions_df = pd.concat(prediction_rows, ignore_index=True)

metrics_out.to_csv(METRICS_CSV, index=False)
rf_importance_summary.to_csv(IMPORTANCE_RF_CSV, index=False)
perm_importance_summary.to_csv(IMPORTANCE_PERM_CSV, index=False)
predictions_df.to_csv(PREDICTIONS_CSV, index=False)

print("\nSaved files:")
print(METRICS_CSV)
print(IMPORTANCE_RF_CSV)
print(IMPORTANCE_PERM_CSV)
print(PREDICTIONS_CSV)