from pathlib import Path
import pandas as pd

# ============================================================
# PREPROCESSING / MERGING SCRIPT
# ============================================================
# This script:
# 1. Loads all feature CSVs
# 2. Harmonizes district ID column names
# 3. Removes NDVI_Mar from the NDVI table
# 4. Loads crop yield CSV
# 5. Builds:
#    - yield_value for 2017-2023
#    - yield_prev using the previous year's observed yield
#      from the full 2016-2023 yield table
# 6. Merges all feature tables on nuts_id + year
# 7. Adds target variable and previous-year yield
# 8. Drops rows without observed yield_value
# 9. Saves the final modelling table
# ============================================================


# ------------------------------------------------------------
# 1) FILE PATHS
# ------------------------------------------------------------

local_dir = Path(
    r"C:\Users\lukas\Documents\Studium\Remote_Sensing_Products\project\crop_yields\data"
)

temp_csv = local_dir / "feature_csv_ger" / "temperature_features_ger_2017_2023.csv"
precip_csv = local_dir / "feature_csv_ger" / "precip_features_ger_2017_2023.csv"
ndvi_csv = local_dir / "feature_csv_ger" / "ndvi_features_ger_2017_2023.csv"
soil_csv = local_dir / "feature_csv_ger" / "soil_moisture_features_ger_2017_2023.csv"

yield_csv = local_dir / "crop_yield" / "Final_data_2024.csv"

output_dir = Path(
    r"C:\Users\lukas\Documents\Studium\Remote_Sensing_Products\project\RSapp_project\Analysis\merged_csv_ger"
)
output_dir.mkdir(parents=True, exist_ok=True)

output_csv = output_dir / "model_table_ww_2017_2023_ger.csv"


# ------------------------------------------------------------
# 2) LOAD FEATURE TABLES
# ------------------------------------------------------------

temp_df = pd.read_csv(temp_csv)
precip_df = pd.read_csv(precip_csv)
ndvi_df = pd.read_csv(ndvi_csv)
soil_df = pd.read_csv(soil_csv)


# ------------------------------------------------------------
# 3) HARMONIZE DISTRICT ID COLUMN
# ------------------------------------------------------------
# Feature CSVs may use NUTS_CODE, while yield data uses nuts_id.
# We standardize everything to nuts_id for merging.
# ------------------------------------------------------------

def harmonize_district_id(df: pd.DataFrame, df_name: str) -> pd.DataFrame:
    if "nuts_id" in df.columns:
        return df
    if "NUTS_CODE" in df.columns:
        return df.rename(columns={"NUTS_CODE": "nuts_id"})
    raise ValueError(f"{df_name} must contain either 'nuts_id' or 'NUTS_CODE'.")


temp_df = harmonize_district_id(temp_df, "temp_df")
precip_df = harmonize_district_id(precip_df, "precip_df")
ndvi_df = harmonize_district_id(ndvi_df, "ndvi_df")
soil_df = harmonize_district_id(soil_df, "soil_df")


# ------------------------------------------------------------
# 4) BASIC CLEANING OF FEATURE TABLES
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
# 5) REMOVE NDVI_Mar
# ------------------------------------------------------------
# NDVI_Mar had missing values in 2017 and is removed.
# ------------------------------------------------------------

if "NDVI_Mar" in ndvi_df.columns:
    ndvi_df = ndvi_df.drop(columns=["NDVI_Mar"])


# ------------------------------------------------------------
# 6) LOAD AND FILTER CROP YIELD DATA
# ------------------------------------------------------------
# We use:
# - 2016-2023 to construct yield_prev
# - then keep 2017-2023 in the final model table
# ------------------------------------------------------------

yield_df = pd.read_csv(yield_csv)

required_cols = {"nuts_id", "year", "var", "measure", "value"}
missing = required_cols - set(yield_df.columns)
if missing:
    raise ValueError(f"Yield CSV is missing required columns: {missing}")

yield_df["nuts_id"] = yield_df["nuts_id"].astype(str).str.strip()
yield_df["year"] = pd.to_numeric(yield_df["year"], errors="coerce").astype("Int64")
yield_df["value"] = pd.to_numeric(yield_df["value"], errors="coerce")

# Keep winter wheat yield data for 2016-2023
yield_df = yield_df[
    (yield_df["year"].between(2016, 2023))
    & (yield_df["var"] == "ww")
    & (yield_df["measure"] == "yield")
].copy()

# Sort for lag construction
yield_df = yield_df.sort_values(["nuts_id", "year"]).reset_index(drop=True)

# Create previous-year yield
yield_df["yield_prev"] = yield_df.groupby("nuts_id")["value"].shift(1)

# Rename target variable
yield_df = yield_df.rename(columns={"value": "yield_value"})

# Keep final modelling years only
yield_df = yield_df[yield_df["year"].between(2017, 2023)].copy()

# Keep only necessary columns
yield_df = yield_df[["nuts_id", "year", "yield_value", "yield_prev"]]


# ------------------------------------------------------------
# 7) OPTIONAL DUPLICATE CHECK
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
# 8) MERGE FEATURE TABLES
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

# Add target variable and previous-year yield
model_df = model_df.merge(
    yield_df,
    on=["nuts_id", "year"],
    how="left"
)


# ------------------------------------------------------------
# 9) DROP ROWS WITHOUT TARGET VARIABLE
# ------------------------------------------------------------
# For modelling, only rows with observed yield_value are kept.
# Missing feature values are NOT dropped here; they are handled
# later in the RF script per model variant.
# ------------------------------------------------------------

rows_before = len(model_df)
model_df = model_df.dropna(subset=["yield_value"]).copy()
rows_after = len(model_df)

print(f"Rows before dropping missing target: {rows_before}")
print(f"Rows after dropping missing target:  {rows_after}")


# ------------------------------------------------------------
# 10) SORT AND SAVE
# ------------------------------------------------------------

model_df = model_df.sort_values(["nuts_id", "year"]).reset_index(drop=True)
model_df.to_csv(output_csv, index=False)

print(f"\nSaved merged model table to:\n{output_csv}")
print(f"Final shape: {model_df.shape}")

print("\nColumns:")
print(model_df.columns.tolist())

print("\nMissing values per column:")
print(model_df.isna().sum()[model_df.isna().sum() > 0].sort_values(ascending=False))

print("\nFirst rows:")
print(model_df.head())