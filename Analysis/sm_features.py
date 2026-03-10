# ============================================================
# SOIL MOISTURE FEATURES FROM ERA5-LAND
# ============================================================
# This script computes soil-moisture-based predictors for crop
# yield modelling at district level.
#
# Data source:
# ERA5-Land Daily Aggregated dataset
#
# Features:
# - SoilMoisture_AprJun:
#   mean absolute soil moisture during April-June
#
# - SoilMoistureAnom_AprJun:
#   anomaly relative to the multi-year baseline mean for the
#   same April-June window
#
# Output:
# One row per district-year with soil moisture predictors.
# ============================================================

import ee

from config import CONFIG
from gee_setup import init_gee, load_districts, get_roi


# ------------------------------------------------------------
# 1) INITIALIZE EARTH ENGINE
# ------------------------------------------------------------

init_gee()


# ------------------------------------------------------------
# 2) LOAD DISTRICTS AND ROI
# ------------------------------------------------------------
# districts:
# one geometry feature per district
#
# roi:
# union geometry of all districts
# ------------------------------------------------------------

districts = load_districts()
roi = get_roi()


# ------------------------------------------------------------
# 3) DEFINE ERA5 DATASET
# ------------------------------------------------------------
# We use ERA5-Land Daily Aggregated and specifically the upper
# soil moisture layer:
#
# volumetric_soil_water_layer_1
#
# This is a coarse-resolution variable (~11 km), so we do not
# apply crop masking here. Aggregation is done directly at
# district level.
# ------------------------------------------------------------

ERA5_DAILY_ID = "ECMWF/ERA5_LAND/DAILY_AGGR"
SOIL_MOISTURE_BAND = "volumetric_soil_water_layer_1"

# Fixed feature window for this script
SOIL_MOISTURE_MONTHS = [4, 5, 6]


# ------------------------------------------------------------
# 4) HELPER FUNCTION: MONTH WINDOW
# ------------------------------------------------------------

def _month_window(year, month):
    """Return [start, end) date window for one month."""
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    return start, end


# ------------------------------------------------------------
# 5) LOAD DAILY SOIL MOISTURE DATA
# ------------------------------------------------------------
# Loads ERA5-Land daily soil moisture images for April-June.
# ------------------------------------------------------------

def _era5_soil_moisture_daily(year):
    """
    Load daily ERA5-Land soil moisture images for the configured
    April-June window.
    """
    months = SOIL_MOISTURE_MONTHS

    start = ee.Date.fromYMD(year, min(months), 1)
    end = ee.Date.fromYMD(year, max(months), 1).advance(1, "month")

    ic = (
        ee.ImageCollection(ERA5_DAILY_ID)
        .filterBounds(roi)
        .filterDate(start, end)
        .select(SOIL_MOISTURE_BAND)
    )

    return ic


# ------------------------------------------------------------
# 6) BUILD SOIL MOISTURE FEATURE IMAGE
# ------------------------------------------------------------
# Features:
# - absolute mean soil moisture in Apr-Jun
# - anomaly relative to baseline years for Apr-Jun
# ------------------------------------------------------------

def build_soil_moisture_feature_image(year):
    """
    Build one multi-band image with soil moisture features for
    a given year.
    """
    # --------------------------------------------------------
    # 6.1 Absolute soil moisture
    # --------------------------------------------------------
    # Mean daily soil moisture during April-June
    # --------------------------------------------------------
    year_ic = _era5_soil_moisture_daily(year)

    soil_moisture_abs = (
        year_ic
        .mean()
        .rename("SoilMoisture_AprJun")
    )

    # --------------------------------------------------------
    # 6.2 Soil moisture anomaly
    # --------------------------------------------------------
    # Anomaly = current year Apr-Jun mean
    #           minus baseline mean over configured years
    # --------------------------------------------------------
    baseline_images = []

    for y in CONFIG["soil_moisture_baseline_years"]:
        baseline_ic = _era5_soil_moisture_daily(y)
        baseline_img = baseline_ic.mean()
        baseline_images.append(baseline_img)

    baseline_mean = ee.ImageCollection.fromImages(baseline_images).mean()

    soil_moisture_anom = (
        soil_moisture_abs
        .subtract(baseline_mean)
        .rename("SoilMoistureAnom_AprJun")
    )

    soil_stack = ee.Image.cat([
        soil_moisture_abs,
        soil_moisture_anom
    ]).clip(roi)

    return soil_stack


# ------------------------------------------------------------
# 7) AGGREGATE TO DISTRICT LEVEL
# ------------------------------------------------------------
# Pixel-based soil moisture features are aggregated to district
# means using reduceRegions().
# ------------------------------------------------------------

def reduce_soil_moisture_to_districts(year):
    """
    Aggregate soil moisture features to district polygons.
    """
    soil_stack = build_soil_moisture_feature_image(year)

    reduced = soil_stack.reduceRegions(
        collection=districts,
        reducer=ee.Reducer.mean(),
        scale=CONFIG["era5_scale"],
    )

    return reduced.map(lambda f: f.set("year", year))


# ------------------------------------------------------------
# 8) BUILD FULL DATASET FOR ALL YEARS
# ------------------------------------------------------------

def build_soil_moisture_table_for_years(years):
    """
    Return one FeatureCollection with all district-year soil
    moisture features.
    """
    merged = ee.FeatureCollection([])

    for y in years:
        yearly_fc = reduce_soil_moisture_to_districts(y)
        merged = merged.merge(yearly_fc)

    return merged


# ------------------------------------------------------------
# 9) SCRIPT ENTRY POINT
# ------------------------------------------------------------
# When run directly, the script:
# 1. builds soil moisture features for all years
# 2. exports the table to Google Drive
# ------------------------------------------------------------

if __name__ == "__main__":

    print("Running soil moisture feature extraction...")

    selectors = [
        CONFIG["district_id_field"],
        "year",
        "SoilMoisture_AprJun",
        "SoilMoistureAnom_AprJun",
    ]

    all_sm_fc = build_soil_moisture_table_for_years(CONFIG["years"])
    all_sm_fc = all_sm_fc.select(selectors)

    task = ee.batch.Export.table.toDrive(
        collection=all_sm_fc,
        description="soil_moisture_features_2017_2023",
        folder=CONFIG["export_folder"],
        fileNamePrefix="soil_moisture_features_2017_2023",
        fileFormat="CSV",
        selectors=selectors
    )

    task.start()

    print("✅ Export started successfully.")
    print("Check the Earth Engine Tasks tab or Google Drive.")