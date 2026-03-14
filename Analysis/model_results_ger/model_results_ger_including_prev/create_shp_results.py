from pathlib import Path
import pandas as pd
import geopandas as gpd

# ============================================================
# JOIN RF PREDICTIONS TO NUTS3 SHAPEFILE
# ============================================================
# This script:
# 1. Loads RF prediction CSV
# 2. Loads NUTS3 shapefile
# 3. Joins CSV predictions to district geometries
#    using:
#       - NUTS_CODE in shapefile
#       - nuts_id   in prediction CSV
# 4. Exports:
#       - one shapefile with all district-year rows
#       - one shapefile per year
# ============================================================


# ------------------------------------------------------------
# 1) FILE PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(
    r"C:\Users\lukas\Documents\Studium\Remote_Sensing_Products\project\RSapp_project"
)

PREDICTIONS_CSV = Path(
    r"C:\Users\lukas\Documents\Studium\Remote_Sensing_Products\project\RSapp_project\Analysis\model_results_ger\model_results_ger_including_prev\rf_leave_one_year_out_predictions_ger_with_prev_yield.csv"
)

SHP_PATH = Path(
    r"C:\Users\lukas\Documents\Studium\Remote_Sensing_Products\project\crop_yields\data\crop_yield\nuts_ger\nuts3_ger\NUTS5000_N3.shp"
)

OUTPUT_DIR = PROJECT_DIR / "Analysis" / "map_outputs_shp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SHP_ALL = OUTPUT_DIR / "rf_pred_nuts3_all.shp"


# ------------------------------------------------------------
# 2) SETTINGS
# ------------------------------------------------------------

CSV_ID_COL = "nuts_id"
SHP_ID_COL = "NUTS_CODE"
YEAR_COL = "year"

# Keep only important prediction columns
# Note: shapefile field names are limited to 10 characters,
# so we rename them below before export.
CSV_COLS_TO_KEEP = [
    "run_name",
    "nuts_id",
    "year",
    "y_true",
    "y_pred",
    "fold_test_year",
    "yield_prev",
]


# ------------------------------------------------------------
# 3) LOAD DATA
# ------------------------------------------------------------

pred_df = pd.read_csv(PREDICTIONS_CSV)
gdf = gpd.read_file(SHP_PATH)

print("\nPrediction CSV columns:")
print(pred_df.columns.tolist())

print("\nShapefile columns:")
print(gdf.columns.tolist())


# ------------------------------------------------------------
# 4) BASIC CHECKS AND CLEANING
# ------------------------------------------------------------

if CSV_ID_COL not in pred_df.columns:
    raise ValueError(f"Prediction CSV must contain '{CSV_ID_COL}'.")

if SHP_ID_COL not in gdf.columns:
    raise ValueError(f"Shapefile must contain '{SHP_ID_COL}'.")

pred_df[CSV_ID_COL] = pred_df[CSV_ID_COL].astype(str).str.strip()
gdf[SHP_ID_COL] = gdf[SHP_ID_COL].astype(str).str.strip()

pred_cols = [c for c in CSV_COLS_TO_KEEP if c in pred_df.columns]
pred_df = pred_df[pred_cols].copy()

pred_df[YEAR_COL] = pd.to_numeric(pred_df[YEAR_COL], errors="coerce").astype("Int64")


# ------------------------------------------------------------
# 5) MERGE PREDICTIONS TO SHAPEFILE
# ------------------------------------------------------------
# This creates one row per district-year.
# The geometry is repeated for each available year.
# ------------------------------------------------------------

gdf_pred = gdf.merge(
    pred_df,
    left_on=SHP_ID_COL,
    right_on=CSV_ID_COL,
    how="left"
)

# keep only rows that actually received a prediction
gdf_pred = gdf_pred.dropna(subset=[YEAR_COL]).copy()

print("\nJoined GeoDataFrame shape:")
print(gdf_pred.shape)

print("\nYears in joined file:")
print(sorted(gdf_pred[YEAR_COL].dropna().unique().tolist()))


# ------------------------------------------------------------
# 6) CREATE RESIDUALS / ERRORS
# ------------------------------------------------------------

if "y_true" in gdf_pred.columns and "y_pred" in gdf_pred.columns:
    gdf_pred["residual"] = gdf_pred["y_true"] - gdf_pred["y_pred"]
    gdf_pred["abs_error"] = (gdf_pred["y_true"] - gdf_pred["y_pred"]).abs()


# ------------------------------------------------------------
# 7) RENAME COLUMNS FOR SHAPEFILE EXPORT
# ------------------------------------------------------------
# Shapefile field names are limited to 10 characters.
# So we shorten the most important columns.
# ------------------------------------------------------------

rename_map = {
    "run_name": "run_name",
    "nuts_id": "nuts_id",
    "year": "year",
    "y_true": "y_true",
    "y_pred": "y_pred",
    "fold_test_year": "fold_year",
    "yield_prev": "y_prev",
    "residual": "resid",
    "abs_error": "abs_err",
}

gdf_pred = gdf_pred.rename(columns=rename_map)


# ------------------------------------------------------------
# 8) EXPORT FULL SHAPEFILE
# ------------------------------------------------------------

gdf_pred.to_file(OUTPUT_SHP_ALL, driver="ESRI Shapefile")
print(f"\nSaved full shapefile to:\n{OUTPUT_SHP_ALL}")


# ------------------------------------------------------------
# 9) EXPORT ONE SHAPEFILE PER YEAR
# ------------------------------------------------------------

years = sorted(gdf_pred["year"].dropna().unique().tolist())

for yr in years:
    gdf_year = gdf_pred[gdf_pred["year"] == yr].copy()

    out_shp = OUTPUT_DIR / f"rf_pred_{int(yr)}.shp"
    gdf_year.to_file(out_shp, driver="ESRI Shapefile")

    print(f"Saved year {int(yr)}: {out_shp}")


print("\nDone.")