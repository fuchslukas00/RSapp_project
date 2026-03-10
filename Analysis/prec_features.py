# ============================================================
# PRECIPITATION FEATURES FROM ERA5-LAND
# ============================================================
# This script computes precipitation-based predictors for crop
# yield modelling at district level.
#
# Data source:
# ERA5-Land Daily Aggregated dataset
#
# Workflow:
# 1. Initialize Google Earth Engine
# 2. Load district geometries
# 3. Load ERA5-Land daily precipitation data
# 4. Compute precipitation-based features
# 5. Aggregate features to district level
# 6. Export results as CSV
#
# Output:
# One row per district-year with precipitation predictors.
# ============================================================

import ee

# Import project configuration and shared setup utilities
from config import CONFIG
from gee_setup import init_gee, load_districts, get_roi


# ------------------------------------------------------------
# 1) INITIALIZE EARTH ENGINE
# ------------------------------------------------------------
# This connects the script to your Google Earth Engine account.
# The project ID is stored centrally in config.py.
# ------------------------------------------------------------

init_gee()


# ------------------------------------------------------------
# 2) LOAD DISTRICT GEOMETRIES
# ------------------------------------------------------------
# These district polygons define the spatial units for which
# precipitation features will later be aggregated.
# ------------------------------------------------------------

districts = load_districts()

# Union of all district geometries
roi = get_roi()


# ------------------------------------------------------------
# 3) DEFINE ERA5 DATASET
# ------------------------------------------------------------
# ERA5-Land Daily Aggregated:
# global reanalysis dataset (~11 km resolution)
#
# Relevant precipitation variable:
# - total_precipitation_sum
#
# Important unit note:
# ERA5-Land daily precipitation is stored in meters of water
# equivalent, so we convert it to millimeters by multiplying
# by 1000.
# ------------------------------------------------------------

ERA5_DAILY_ID = "ECMWF/ERA5_LAND/DAILY_AGGR"


# ------------------------------------------------------------
# 4) HELPER FUNCTION: MONTH WINDOW
# ------------------------------------------------------------
# Returns start and end date of a given month.
#
# Example:
# month_window(2020, 3)
# -> 2020-03-01 to 2020-04-01
# ------------------------------------------------------------

def _month_window(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    return start, end


# ------------------------------------------------------------
# 5) LOAD DAILY PRECIPITATION DATA
# ------------------------------------------------------------
# Loads ERA5-Land daily images for the configured precipitation
# months and converts precipitation from meters to millimeters.
#
# Also clamps negative values to zero because ERA5 aggregated
# precipitation can occasionally contain tiny negative artifacts.
# ------------------------------------------------------------

def _era5_precip_daily(year):
    """
    Load daily ERA5-Land precipitation images for the configured
    monthly precipitation window and add precipitation in mm.
    """
    months = CONFIG["monthly_precip_months"]

    start = ee.Date.fromYMD(year, min(months), 1)
    end = ee.Date.fromYMD(year, max(months), 1).advance(1, "month")

    ic = (
        ee.ImageCollection(ERA5_DAILY_ID)
        .filterBounds(roi)
        .filterDate(start, end)
    )

    def add_precip_mm(img):
        precip_mm = (
            img.select("total_precipitation_sum")
            .max(0)                 # remove small negative artifacts
            .multiply(1000.0)       # meters -> millimeters
            .rename("Precip_mm")
        )
        return img.addBands(precip_mm)

    return ic.map(add_precip_mm)


# ------------------------------------------------------------
# 6) BUILD PRECIPITATION FEATURE IMAGE
# ------------------------------------------------------------
# For one year this function builds a multi-band image
# containing all required precipitation predictors.
#
# Features:
# - monthly precipitation sums
# - count of rainy days above threshold
# ------------------------------------------------------------

def build_precip_feature_image(year):
    """
    Build one multi-band image with all required precipitation
    features for a given year.
    """
    daily = _era5_precip_daily(year)

    # --------------------------------------------------------
    # 6.1 Monthly precipitation sums
    # --------------------------------------------------------
    # For each configured month:
    # sum all daily precipitation values in mm
    # --------------------------------------------------------

    month_name_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }

    monthly_sum_bands = []

    for month in CONFIG["monthly_precip_months"]:
        month_start, month_end = _month_window(year, month)

        month_ic = daily.filterDate(month_start, month_end)

        month_label = month_name_map[month]

        precip_sum = (
            month_ic
            .select("Precip_mm")
            .sum()
            .rename(f"PrecipSum_{month_label}")
        )

        monthly_sum_bands.append(precip_sum)

    # --------------------------------------------------------
    # 6.2 Count of rainy days for the full season (Mar–Jun)
    # --------------------------------------------------------
    # A rainy day is defined as:
    # precipitation >= CONFIG["rainy_day_threshold_mm"]
    #
    # Instead of computing this per month, we aggregate over
    # the full March–June period.
    # --------------------------------------------------------

    rainy_threshold = CONFIG["rainy_day_threshold_mm"]

    season_months = CONFIG["monthly_precip_months"]

    season_start = ee.Date.fromYMD(year, min(season_months), 1)
    season_end = ee.Date.fromYMD(year, max(season_months), 1).advance(1, "month")

    season_ic = daily.filterDate(season_start, season_end)

    def rainy_day(img):
        return (
            img.select("Precip_mm")
            .gte(rainy_threshold)
            .rename("RainyDay")
        )

    rainy_days_season = (
        season_ic
        .map(rainy_day)
        .sum()
        .rename(f"RainyDays_>={int(rainy_threshold)}mm_Season")
    )

    # --------------------------------------------------------
    # 6.3 Combine all predictors into one image
    # --------------------------------------------------------

    precip_stack = ee.Image.cat(
        monthly_sum_bands + [rainy_days_season]
    ).clip(roi)

    return precip_stack


# ------------------------------------------------------------
# 7) AGGREGATE FEATURES TO DISTRICT LEVEL
# ------------------------------------------------------------
# Converts pixel-based precipitation predictors into district
# statistics.
#
# Operation:
# reduceRegions()
#
# For each district polygon:
# mean value of each predictor band is calculated.
# ------------------------------------------------------------

def reduce_precip_to_districts(year, use_crop_mask=False):
    """
    Aggregate precipitation features to district polygons.

    Args:
        year: processing year
        use_crop_mask: if True, aggregate only over masked crop pixels
                       (not enabled yet)
    """
    precip_stack = build_precip_feature_image(year)

    if use_crop_mask:
        raise NotImplementedError("Crop mask support not enabled yet.")

    reduced = precip_stack.reduceRegions(
        collection=districts,
        reducer=ee.Reducer.mean(),
        scale=CONFIG["era5_scale"],
    )

    return reduced.map(lambda f: f.set("year", year))


# ------------------------------------------------------------
# 8) BUILD FULL DATASET (ALL YEARS)
# ------------------------------------------------------------
# Loops over all configured years and merges district-year
# results into one FeatureCollection.
# ------------------------------------------------------------

def build_precip_table_for_years(years, use_crop_mask=False):
    """
    Return one FeatureCollection with all district-year
    precipitation features.
    """
    merged = ee.FeatureCollection([])

    for y in years:
        yearly_fc = reduce_precip_to_districts(y, use_crop_mask=use_crop_mask)
        merged = merged.merge(yearly_fc)

    return merged


# ------------------------------------------------------------
# 9) SCRIPT ENTRY POINT
# ------------------------------------------------------------
# When this script is run directly, it:
#
# 1. Builds the precipitation feature table
# 2. Exports results to Google Drive
# ------------------------------------------------------------

if __name__ == "__main__":

    print("Running precipitation feature extraction...")

    # Quick check for one example year
    sample_year = CONFIG["years"][0]
    sample_precip_image = build_precip_feature_image(sample_year)

    print("Precipitation feature bands:")
    print(sample_precip_image.bandNames().getInfo())

    sample_district_table = reduce_precip_to_districts(sample_year, use_crop_mask=False)
    print(f"District rows for {sample_year}: {sample_district_table.size().getInfo()}")

    # Define the export columns explicitly
    selectors = [
        CONFIG["district_id_field"],
        "year",
        "PrecipSum_Mar",
        "PrecipSum_Apr",
        "PrecipSum_May",
        "PrecipSum_Jun",
        "RainyDays_>=1mm_Season",
    ]

    all_precip_fc = build_precip_table_for_years(CONFIG["years"], use_crop_mask=False)
    all_precip_fc = all_precip_fc.select(selectors)

    task = ee.batch.Export.table.toDrive(
        collection=all_precip_fc,
        description="precip_features_2017_2023",
        folder=CONFIG["export_folder"],
        fileNamePrefix="precip_features_2017_2023",
        fileFormat="CSV",
        selectors=selectors
    )

    task.start()

    print("✅ Export started successfully.")
    print("Check the Earth Engine Tasks tab or Google Drive.")