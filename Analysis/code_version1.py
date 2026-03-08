# ============================================================
# STEP 1: INITIALIZATION AND INPUT DEFINITIONS
# ============================================================
# Purpose of this block:
# 1. Import the required Python package
# 2. Initialize the Google Earth Engine Python API
# 3. Define all user-specific input paths and key parameters
# 4. Load the main spatial input datasets:
#    - district boundaries
#    - crop mask collection
# 5. Run a few basic checks so we know the setup works
#
# This block does NOT yet compute any predictors.
# It only prepares the project environment.
# ============================================================

import ee
import pandas as pd
import json
from datetime import datetime

# ------------------------------------------------------------
# 1) INITIALIZE EARTH ENGINE
# ------------------------------------------------------------
# Earth Engine requires:
# - prior authentication via:
#     earthengine authenticate
# - a valid Google Cloud project
#
# Replace the project name below with your own GCP project ID.
# If initialization works, the API is ready for use.
# ------------------------------------------------------------

ee.Initialize(project="rsapp-25-26")


# ------------------------------------------------------------
# 2) CENTRAL PROJECT CONFIGURATION
# ------------------------------------------------------------
# All important user-defined settings are stored in one place.
# This makes the script easier to adapt and helps avoid
# hard-coded paths spread throughout the code.
#
# You will later adjust:
# - district asset path
# - crop mask asset path
# - attribute names
# - target years
# - season windows
# ------------------------------------------------------------

CONFIG = {
    # --------------------------------------------------------
    # Spatial boundary dataset: districts / NUTS3 / counties
    # --------------------------------------------------------
    # This should point to a FeatureCollection uploaded to GEE.
    # Example:
    # "projects/your-project/assets/germany_districts"
    #
    # In this project the same asset contains:
    # - district geometry
    # - yield/area observations per district-year
    #
    # Required columns in that asset:
    # - nuts_id
    # - measure (yield or area)
    # - year
    # - value
    # --------------------------------------------------------
    "districts_asset": "projects/rsapp-25-26/assets/district_boundaries",

    # Name of the unique district ID field in the boundary layer.
    # Example: "NUTS_ID", "district_id", "LK_ID"
    "district_id_field": "nuts_id",

    # Optional district name field.
    # Set to None if your asset has no readable name column.
    "district_name_field": None,

    # Yield table fields inside the same boundary asset.
    "yield_measure_field": "measure",
    "yield_year_field": "year",
    "yield_value_field": "value",
    "yield_measure_values": {
        "yield": "yield",
        "area": "area",
    },

    # --------------------------------------------------------
    # Crop mask dataset
    # --------------------------------------------------------
    # This is assumed to be an ImageCollection where:
    # - each image represents one year
    # - each image has a property called "year"
    # - one band stores the crop class
    #
    # Example:
    # "projects/your-project/assets/cropmask_wheat_by_year"
    # --------------------------------------------------------
    "crop_mask_asset": "projects/rsapp-25-26/assets/cropmask_targetcrop_by_year",

    # Name of the crop-class band in each image
    "crop_mask_band": "crop_mask",

    # Pixel value representing the target crop
    # Example:
    # 1 = target crop, 0 = other
    "crop_mask_value": 1,

    # --------------------------------------------------------
    # Temporal setup
    # --------------------------------------------------------
    # Years to process.
    # Adjust depending on availability of:
    # - crop mask
    # - Sentinel-2
    # - yield data
    # --------------------------------------------------------
    "years": list(range(2017, 2024)),

    # --------------------------------------------------------
    # Season windows
    # --------------------------------------------------------
    # These are placeholder values for now.
    # They will later be justified in the paper/methodology.
    #
    # Example here for winter wheat:
    # March-July main active season
    # --------------------------------------------------------
    "season_months": [3, 4, 5, 6, 7],
    "early_months": [3, 4],
    "mid_months": [5, 6],
    "late_months": [7],

    # --------------------------------------------------------
    # Climate thresholds
    # --------------------------------------------------------
    # These values are also methodological choices and should
    # later be justified with literature.
    # --------------------------------------------------------
    "gdd_base_temp_c": 5.0,
    "hot_day_threshold_c": 30.0,
    "rainy_day_threshold_mm": 1.0,

    # --------------------------------------------------------
    # Soil moisture anomaly baseline years
    # --------------------------------------------------------
    # Used later if soil moisture anomaly is computed.
    # The baseline period should be explained in the paper.
    # --------------------------------------------------------
    "soil_moisture_baseline_years": list(range(2017, 2024)),

    # --------------------------------------------------------
    # Sentinel-2 cloud masking parameter
    # --------------------------------------------------------
    "s2_cloud_probability_threshold": 40,

    # --------------------------------------------------------
    # Suggested spatial scales
    # --------------------------------------------------------
    # These are not used yet, but defined here so that all
    # resolution choices live in one central place.
    # --------------------------------------------------------
    "s2_scale": 10,
    "era5_scale": 11132,

    # --------------------------------------------------------
    # Export settings
    # --------------------------------------------------------
    "export_description": "district_predictors_export",
    "export_folder": "GEE_exports",
    "export_file_prefix": "district_predictors"
}


# ------------------------------------------------------------
# 3) LOAD DISTRICT BOUNDARIES
# ------------------------------------------------------------
# The uploaded asset includes both geometry and tabular yield rows.
# We keep two views:
# - districts_raw: full feature set (for yield/area filtering later)
# - districts: one geometry feature per nuts_id for climate reduction
# ------------------------------------------------------------

districts_raw = ee.FeatureCollection(CONFIG["districts_asset"])

district_selectors = [CONFIG["district_id_field"]]
if CONFIG["district_name_field"]:
    district_selectors.append(CONFIG["district_name_field"])

districts = districts_raw.select(district_selectors).distinct([CONFIG["district_id_field"]])

# Geometry of the whole study area (union of all districts)
roi = districts.geometry()


# ------------------------------------------------------------
# 4) LOAD CROP MASK COLLECTION
# ------------------------------------------------------------
# TEMPORARILY DISABLED: Crop mask assets not yet uploaded.
# Will be enabled once cropmask_targetcrop_by_year is available.
# --------------------------------------------------------

# crop_mask_collection = ee.ImageCollection(CONFIG["crop_mask_asset"])


# ------------------------------------------------------------
# 5) BASIC SANITY CHECKS
# ------------------------------------------------------------
# These checks are useful to confirm that:
# - the assets are reachable
# - the collections are not empty
# - the script is connected correctly to GEE
#
# Note:
# getInfo() sends a request to the server and returns results
# to Python. Use it sparingly in larger workflows, but it is
# very useful during setup and debugging.
# ------------------------------------------------------------

print("Earth Engine initialized successfully.")

# Number of rows in full combined table
districts_raw_count = districts_raw.size().getInfo()
print(f"Rows in combined boundary+yield asset: {districts_raw_count}")

# Number of unique district geometries
district_count = districts.size().getInfo()
print(f"Number of unique district geometries: {district_count}")

# Number of crop mask images
# (temporarily disabled - crop mask assets not yet available)
# crop_mask_count = crop_mask_collection.size().getInfo()
# print(f"Number of crop mask images: {crop_mask_count}")

# Print the configured years
print("Configured years:", CONFIG["years"])

# Print the season windows
print("Season months:", CONFIG["season_months"])
print("Early season months:", CONFIG["early_months"])
print("Mid season months:", CONFIG["mid_months"])
print("Late season months:", CONFIG["late_months"])

# Distinct measure values available in the boundary asset
available_measures = districts_raw.aggregate_array(CONFIG["yield_measure_field"]).distinct().getInfo()
print("Available measure values:", available_measures)


# ------------------------------------------------------------
# 6) OPTIONAL: INSPECT THE FIRST DISTRICT
# ------------------------------------------------------------
# This helps confirm that the selected ID/name fields are present.
# ------------------------------------------------------------

first_district = districts.first().getInfo()
print("\nExample district feature:")
print(first_district)


# ------------------------------------------------------------
# 7) OPTIONAL: INSPECT ONE YIELD/AREA ROW FROM THE SAME ASSET
# ------------------------------------------------------------

first_yield_row = districts_raw.select(
    [
        CONFIG["district_id_field"],
        CONFIG["yield_measure_field"],
        CONFIG["yield_year_field"],
        CONFIG["yield_value_field"],
    ]
).first().getInfo()
print("\nExample yield/area row from boundary asset:")
print(first_yield_row)


# ------------------------------------------------------------
# 8) OPTIONAL: INSPECT THE FIRST CROP MASK IMAGE
# --------------------------------------------------------
# (temporarily disabled - crop mask assets not yet available)
# This helps check whether:
# - the collection contains images
# - the "year" property exists
# - the expected crop mask band exists
# --------------------------------------------------------

# first_crop_mask = crop_mask_collection.first().getInfo()
# print("\nExample crop mask image metadata:")
# print(first_crop_mask)


def get_observed_table(measure_key):
    """
    Return observed rows (yield or area) from the combined boundary asset.

    Args:
        measure_key: "yield" or "area" (keys from CONFIG["yield_measure_values"]).
    """
    if measure_key not in CONFIG["yield_measure_values"]:
        raise ValueError(
            f"Unknown measure_key '{measure_key}'. "
            f"Use one of: {list(CONFIG['yield_measure_values'].keys())}"
        )

    measure_value = CONFIG["yield_measure_values"][measure_key]

    return districts_raw.filter(
        ee.Filter.eq(CONFIG["yield_measure_field"], measure_value)
    ).select(
        [
            CONFIG["district_id_field"],
            CONFIG["yield_measure_field"],
            CONFIG["yield_year_field"],
            CONFIG["yield_value_field"],
        ]
    )


# ============================================================
# STEP 2: ERA5-LAND TEMPERATURE FEATURES (DISTRICT LEVEL)
# ============================================================
# Ziel-Features:
# - GDD_Mar_Jun
# - HeatDays_>27C_May_Jun
# - Median_Tmax_Apr, Median_Tmax_May, Median_Tmax_Jun
# - Median_Tmin_Apr, Median_Tmin_May, Median_Tmin_Jun
#
# Vorgehen:
# 1) Tagesdaten aus ERA5-Land (Kelvin) laden
# 2) In Celsius umrechnen
# 3) Feature-Bilder pro Jahr bilden
# 4) Auf District-Polygone aggregieren
# ============================================================

ERA5_DAILY_ID = "ECMWF/ERA5_LAND/DAILY_AGGR"


def _month_window(year, month):
    """Return [start, end) date window for one month."""
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    return start, end


def _era5_temperature_daily(year):
    """
    Load daily ERA5-Land images for Mar-Jun and add temperature bands in Celsius.

    Important:
    - Input bands are Kelvin in ERA5-Land.
    - Output bands here are Celsius.
    """
    start = ee.Date.fromYMD(year, 3, 1)
    end = ee.Date.fromYMD(year, 7, 1)  # exclusive -> includes March to June

    ic = (
        ee.ImageCollection(ERA5_DAILY_ID)
        .filterBounds(roi)
        .filterDate(start, end)
    )

    def add_celsius_bands(img):
        tmax_c = img.select("temperature_2m_max").subtract(273.15).rename("Tmax_C")
        tmin_c = img.select("temperature_2m_min").subtract(273.15).rename("Tmin_C")
        tmean_c = tmax_c.add(tmin_c).divide(2).rename("Tmean_C")
        return img.addBands([tmax_c, tmin_c, tmean_c])

    return ic.map(add_celsius_bands)


def build_temperature_feature_image(year):
    """Build one multi-band image with all required temperature features."""
    daily = _era5_temperature_daily(year)

    # 1) GDD March-June: sum(max(Tmean - base, 0))
    base_temp_c = CONFIG["gdd_base_temp_c"]

    def gdd_day(img):
        return img.select("Tmean_C").subtract(base_temp_c).max(0).rename("GDD_day")

    gdd_mar_jun = daily.map(gdd_day).sum().rename("GDD_Mar_Jun")

    # 2) Heat days > 27C in May-June (based on Tmax)
    may_start = ee.Date.fromYMD(year, 5, 1)
    jul_start = ee.Date.fromYMD(year, 7, 1)
    may_jun = daily.filterDate(may_start, jul_start)

    def heat_day(img):
        return img.select("Tmax_C").gt(27).rename("HeatDay")

    heat_days_may_jun = may_jun.map(heat_day).sum().rename("HeatDays_>27C_May_Jun")

    # 3-8) Monthly medians of Tmax and Tmin for Apr/May/Jun
    apr_start, apr_end = _month_window(year, 4)
    may_start, may_end = _month_window(year, 5)
    jun_start, jun_end = _month_window(year, 6)

    apr = daily.filterDate(apr_start, apr_end)
    may = daily.filterDate(may_start, may_end)
    jun = daily.filterDate(jun_start, jun_end)

    median_tmax_apr = apr.select("Tmax_C").median().rename("Median_Tmax_Apr")
    median_tmax_may = may.select("Tmax_C").median().rename("Median_Tmax_May")
    median_tmax_jun = jun.select("Tmax_C").median().rename("Median_Tmax_Jun")

    median_tmin_apr = apr.select("Tmin_C").median().rename("Median_Tmin_Apr")
    median_tmin_may = may.select("Tmin_C").median().rename("Median_Tmin_May")
    median_tmin_jun = jun.select("Tmin_C").median().rename("Median_Tmin_Jun")

    temp_stack = ee.Image.cat([
        gdd_mar_jun,
        heat_days_may_jun,
        median_tmax_apr,
        median_tmax_may,
        median_tmax_jun,
        median_tmin_apr,
        median_tmin_may,
        median_tmin_jun,
    ]).clip(roi)

    return temp_stack


def reduce_temperature_to_districts(year, use_crop_mask=False):
    """
    Aggregate temperature features to district polygons.

    Args:
        year: processing year
        use_crop_mask: if True, aggregate only over masked crop pixels
    """
    temp_stack = build_temperature_feature_image(year)

    if use_crop_mask:
        # Re-use yearly crop mask from the uploaded collection.
        crop_mask = (
            crop_mask_collection
            .filter(ee.Filter.eq("year", year))
            .first()
        )
        crop_mask = ee.Image(crop_mask).select(CONFIG["crop_mask_band"]).eq(CONFIG["crop_mask_value"]).selfMask()
        temp_stack = temp_stack.updateMask(crop_mask)

    reduced = temp_stack.reduceRegions(
        collection=districts,
        reducer=ee.Reducer.mean(),
        scale=CONFIG["era5_scale"],
    )

    return reduced.map(lambda f: f.set("year", year))


def build_temperature_table_for_years(years, use_crop_mask=False):
    """Return one FeatureCollection with all district-year temperature features."""
    merged = ee.FeatureCollection([])

    for y in years:
        yearly_fc = reduce_temperature_to_districts(y, use_crop_mask=use_crop_mask)
        merged = merged.merge(yearly_fc)

    return merged


# ------------------------------------------------------------
# 8) QUICK CHECK FOR ONE YEAR
# ------------------------------------------------------------
sample_year = CONFIG["years"][0]

sample_temp_image = build_temperature_feature_image(sample_year)
print("\nTemperature feature bands:")
print(sample_temp_image.bandNames().getInfo())

sample_district_table = reduce_temperature_to_districts(sample_year, use_crop_mask=False)
print(f"District rows for {sample_year}:", sample_district_table.size().getInfo())

sample_row = sample_district_table.first().getInfo()
print("\nExample district temperature row:")
print(sample_row)


# ============================================================
# STEP 3: EXPORT ALL TEMPERATURE FEATURES TO CSV
# ============================================================

print("\n" + "="*60)
print("EXPORTING TEMPERATURE FEATURES TO CSV")
print("="*60)

# Build temperature table for all configured years
print(f"Building temperature table for years: {CONFIG['years']}")
all_temp_fc = build_temperature_table_for_years(CONFIG["years"], use_crop_mask=False)

# Optional: keep only relevant columns
selectors = [
    "nuts_id",
    "year",
    "GDD_Mar_Jun",
    "HeatDays_>27C_May_Jun",
    "Median_Tmax_Apr",
    "Median_Tmax_May",
    "Median_Tmax_Jun",
    "Median_Tmin_Apr",
    "Median_Tmin_May",
    "Median_Tmin_Jun",
]

all_temp_fc = all_temp_fc.select(selectors)

# Export directly from Earth Engine to Google Drive
task = ee.batch.Export.table.toDrive(
    collection=all_temp_fc,
    description="temperature_features_2017_2023",
    folder="GEE_exports",
    fileNamePrefix="temperature_features_2017_2023",
    fileFormat="CSV",
    selectors=selectors
)

task.start()

print("✅ Export task started successfully.")
print("Check the Earth Engine Tasks tab or Google Drive folder 'GEE_exports'.")