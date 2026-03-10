from pathlib import Path
import pandas as pd

# ============================================================
# PREPROCESSING SCRIPT
# ============================================================
# This script:
# 1. Loads all feature CSVs
# 2. Removes NDVI_Mar from the NDVI table
# 3. Loads and filters the crop yield CSV
#    - years 2017-2023
#    - winter wheat only (var == "ww")
#    - yield only (measure == "yield")
# 4. Merges all feature tables on nuts_id + year
# 5. Adds the observed yield value as target variable
# 6. Saves the final modelling table
# ============================================================

# ------------------------------------------------------------
# 1) FILE PATHS
# ------------------------------------------------------------

# csv data path
project_dir = Path(r"C:\Users\lukas\Documents\Studium\Remote_Sensing_Products\project\crop_yields\data")


temp_csv = project_dir / "feature_csv" / "temperature_features_2017_2023.csv"
precip_csv = project_dir / "feature_csv" / "precip_features_2017_2023.csv"
ndvi_csv = project_dir / "feature_csv" / "ndvi_features_2017_2023.csv"
soil_csv = project_dir / "feature_csv" / "soil_moisture_features_2017_2023.csv"

yield_csv = Path(
    r"C:\Users\lukas\Documents\Studium\Remote_Sensing_Products\project\crop_yields\data\crop_yield\Final_data_2024.csv"
)

output_dir = project_dir / "feature_csv" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_csv = output_dir / "model_table_ww_2017_2023.csv"


# ------------------------------------------------------------
# 2) LOAD FEATURE TABLES
# ------------------------------------------------------------

temp_df = pd.read_csv(temp_csv)
precip_df = pd.read_csv(precip_csv)
ndvi_df = pd.read_csv(ndvi_csv)
soil_df = pd.read_csv(soil_csv)


# ------------------------------------------------------------
# 3) BASIC CLEANING OF FEATURE TABLES
# ------------------------------------------------------------
# Make sure join keys are consistent across all tables
# ------------------------------------------------------------

feature_tables = {
    "temp": temp_df,
    "precip": precip_df,
    "ndvi": ndvi_df,
    "soil": soil_df,
}

for name, df in feature_tables.items():
    if "nuts_id" not in df.columns or "year" not in df.columns:
        raise ValueError(f"{name} table must contain 'nuts_id' and 'year'")

    df["nuts_id"] = df["nuts_id"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")


# ------------------------------------------------------------
# 4) REMOVE NDVI_Mar
# ------------------------------------------------------------
# NDVI_Mar had missing values in 2017 and is therefore removed
# from the predictor set.
# ------------------------------------------------------------

if "NDVI_Mar" in ndvi_df.columns:
    ndvi_df = ndvi_df.drop(columns=["NDVI_Mar"])


# ------------------------------------------------------------
# 5) LOAD AND FILTER CROP YIELD DATA
# ------------------------------------------------------------
# Keep only:
# - years 2017 to 2023
# - winter wheat (var == "ww")
# - yield rows only (measure == "yield")
# ------------------------------------------------------------

yield_df = pd.read_csv(yield_csv)

required_cols = {"nuts_id", "year", "var", "measure", "value"}
missing = required_cols - set(yield_df.columns)
if missing:
    raise ValueError(f"Yield CSV is missing required columns: {missing}")

yield_df["nuts_id"] = yield_df["nuts_id"].astype(str).str.strip()
yield_df["year"] = pd.to_numeric(yield_df["year"], errors="coerce").astype("Int64")
yield_df["value"] = pd.to_numeric(yield_df["value"], errors="coerce")

yield_df = yield_df[
    (yield_df["year"].between(2017, 2023))
    & (yield_df["var"] == "ww")
    & (yield_df["measure"] == "yield")
].copy()

# Rename target variable for clarity
yield_df = yield_df.rename(columns={"value": "yield_value"})

# Keep only the columns needed for merging
yield_df = yield_df[["nuts_id", "year", "yield_value"]]


# ------------------------------------------------------------
# 6) OPTIONAL DUPLICATE CHECK
# ------------------------------------------------------------
# Ideally there should be exactly one row per nuts_id-year
# in each table.
# ------------------------------------------------------------

def report_duplicates(df: pd.DataFrame, name: str) -> None:
    n_dupes = df.duplicated(subset=["nuts_id", "year"]).sum()
    if n_dupes > 0:
        print(f"WARNING: {name} has {n_dupes} duplicate nuts_id-year rows.")

report_duplicates(temp_df, "temp_df")
report_duplicates(precip_df, "precip_df")
report_duplicates(ndvi_df, "ndvi_df")
report_duplicates(soil_df, "soil_df")
report_duplicates(yield_df, "yield_df")


# ------------------------------------------------------------
# 7) MERGE FEATURE TABLES
# ------------------------------------------------------------
# We use the feature tables as the basis.
# This means the final dataset will only contain district-year
# combinations that exist in the feature CSVs.
#
# If your feature CSVs only cover Baden-Württemberg, then the
# final merged table will also only cover Baden-Württemberg.
# ------------------------------------------------------------

model_df = temp_df.merge(
    precip_df,
    on=["nuts_id", "year"],
    how="left"
)

model_df = model_df.merge(
    ndvi_df,
    on=["nuts_id", "year"],
    how="left"
)

model_df = model_df.merge(
    soil_df,
    on=["nuts_id", "year"],
    how="left"
)

# Add target variable from yield CSV
model_df = model_df.merge(
    yield_df,
    on=["nuts_id", "year"],
    how="left"
)


# ------------------------------------------------------------
# 8) DROP ROWS WITHOUT TARGET VARIABLE
# ------------------------------------------------------------
# For modelling, we only keep rows with an observed yield.
# ------------------------------------------------------------

rows_before = len(model_df)
model_df = model_df.dropna(subset=["yield_value"]).copy()
rows_after = len(model_df)

print(f"Rows before dropping missing target: {rows_before}")
print(f"Rows after dropping missing target:  {rows_after}")


# ------------------------------------------------------------
# 9) SORT AND SAVE
# ------------------------------------------------------------

model_df = model_df.sort_values(["nuts_id", "year"]).reset_index(drop=True)
model_df.to_csv(output_csv, index=False)

print(f"\nSaved merged model table to:\n{output_csv}")
print(f"Final shape: {model_df.shape}")

print("\nColumns:")
print(model_df.columns.tolist())

print("\nFirst rows:")
print(model_df.head())