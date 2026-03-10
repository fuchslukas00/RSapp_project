# ============================================================
# TEMPERATURE FEATURES FROM ERA5-LAND
# ============================================================
# This script computes temperature-based predictors for crop
# yield modelling at district level.
#
# Data source:
# ERA5-Land Daily Aggregated dataset
#
# Workflow:
# 1. Initialize Google Earth Engine
# 2. Load district geometries
# 3. Load ERA5-Land daily temperature data
# 4. Compute temperature-based features
# 5. Aggregate features to district level
# 6. Export results as CSV
#
# Output:
# One row per district-year with temperature predictors.
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
# climate features will later be aggregated.
#
# Example:
# NUTS3 districts in Germany
# ------------------------------------------------------------

districts = load_districts()

# Union of all district geometries.
# Used to spatially filter satellite datasets.
roi = get_roi()


# ------------------------------------------------------------
# 3) DEFINE ERA5 DATASET
# ------------------------------------------------------------
# ERA5-Land Daily Aggregated:
# global reanalysis dataset (~11 km resolution)
#
# Key temperature variables:
# - temperature_2m_max
# - temperature_2m_min
#
# Units: Kelvin
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
# 5) LOAD DAILY TEMPERATURE DATA
# ------------------------------------------------------------
# Loads ERA5-Land images for the months defined in CONFIG.
#
# Steps:
# 1. Filter ERA5 data by date
# 2. Convert Kelvin -> Celsius
# 3. Compute daily mean temperature
# ------------------------------------------------------------

def _era5_temperature_daily(year):

    months = CONFIG["gdd_months"]

    # Start and end date for the climate window
    start = ee.Date.fromYMD(year, min(months), 1)
    end = ee.Date.fromYMD(year, max(months), 1).advance(1, "month")

    ic = (
        ee.ImageCollection(ERA5_DAILY_ID)
        .filterBounds(roi)
        .filterDate(start, end)
    )

    # Convert Kelvin to Celsius and compute mean temperature
    def add_celsius_bands(img):

        # Maximum daily temperature
        tmax_c = (
            img.select("temperature_2m_max")
            .subtract(273.15)
            .rename("Tmax_C")
        )

        # Minimum daily temperature
        tmin_c = (
            img.select("temperature_2m_min")
            .subtract(273.15)
            .rename("Tmin_C")
        )

        # Daily mean temperature
        tmean_c = (
            tmax_c.add(tmin_c)
            .divide(2)
            .rename("Tmean_C")
        )

        return img.addBands([tmax_c, tmin_c, tmean_c])

    return ic.map(add_celsius_bands)


# ------------------------------------------------------------
# 6) BUILD TEMPERATURE FEATURE IMAGE
# ------------------------------------------------------------
# For one year this function builds a multi-band image
# containing all temperature predictors.
#
# Features:
# - Growing Degree Days
# - Heat days above threshold
# - Monthly median Tmax
# - Monthly median Tmin
# ------------------------------------------------------------

def build_temperature_feature_image(year):

    daily = _era5_temperature_daily(year)

    # --------------------------------------------------------
    # 6.1 Growing Degree Days (GDD)
    # --------------------------------------------------------
    # GDD = sum(max(Tmean - base_temp, 0))
    #
    # Proxy for accumulated thermal energy for crop growth.
    # --------------------------------------------------------

    base_temp_c = CONFIG["gdd_base_temp_c"]

    def gdd_day(img):
        return (
            img.select("Tmean_C")
            .subtract(base_temp_c)
            .max(0)
            .rename("GDD_day")
        )

    gdd_months = CONFIG["gdd_months"]
    gdd_label = f"{min(gdd_months):02d}_{max(gdd_months):02d}"

    gdd_sum = (
        daily
        .map(gdd_day)
        .sum()
        .rename(f"GDD_{gdd_label}")
    )


    # --------------------------------------------------------
    # 6.2 Heat stress days
    # --------------------------------------------------------
    # Count number of days where Tmax exceeds threshold.
    #
    # Example:
    # Tmax > 27°C during flowering or grain filling.
    # --------------------------------------------------------

    heat_months = CONFIG["heat_months"]

    heat_start = ee.Date.fromYMD(year, min(heat_months), 1)
    heat_end = ee.Date.fromYMD(year, max(heat_months), 1).advance(1, "month")

    heat_period = daily.filterDate(heat_start, heat_end)

    threshold = CONFIG["hot_day_threshold_c"]

    def heat_day(img):
        return img.select("Tmax_C").gt(threshold).rename("HeatDay")

    heat_days = (
        heat_period
        .map(heat_day)
        .sum()
        .rename(f"HeatDays_>{int(threshold)}C_{min(heat_months):02d}_{max(heat_months):02d}")
    )


    # --------------------------------------------------------
    # 6.3 Monthly temperature statistics
    # --------------------------------------------------------
    # Median Tmax and Tmin for selected months.
    #
    # These capture temperature conditions during
    # different phenological stages.
    # --------------------------------------------------------

    month_name_map = {
        1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
        7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
    }

    monthly_bands = []

    for month in CONFIG["monthly_temp_months"]:

        month_start, month_end = _month_window(year, month)

        month_ic = daily.filterDate(month_start, month_end)

        month_label = month_name_map[month]

        median_tmax = (
            month_ic
            .select("Tmax_C")
            .median()
            .rename(f"Median_Tmax_{month_label}")
        )

        median_tmin = (
            month_ic
            .select("Tmin_C")
            .median()
            .rename(f"Median_Tmin_{month_label}")
        )

        monthly_bands.extend([median_tmax, median_tmin])


    # --------------------------------------------------------
    # 6.4 Combine all predictors into one image
    # --------------------------------------------------------

    temp_stack = ee.Image.cat(
        [gdd_sum, heat_days] + monthly_bands
    ).clip(roi)

    return temp_stack


# ------------------------------------------------------------
# 7) AGGREGATE FEATURES TO DISTRICT LEVEL
# ------------------------------------------------------------
# Converts pixel-based predictors into district statistics.
#
# Operation:
# reduceRegions()
#
# For each district polygon:
# mean value of each predictor band is calculated.
# ------------------------------------------------------------

def reduce_temperature_to_districts(year, use_crop_mask=False):

    temp_stack = build_temperature_feature_image(year)

    reduced = temp_stack.reduceRegions(
        collection=districts,
        reducer=ee.Reducer.mean(),
        scale=CONFIG["era5_scale"],
    )

    return reduced.map(lambda f: f.set("year", year))


# ------------------------------------------------------------
# 8) BUILD FULL DATASET (ALL YEARS)
# ------------------------------------------------------------
# Loops over all years and merges results into one table.
# ------------------------------------------------------------

def build_temperature_table_for_years(years):

    merged = ee.FeatureCollection([])

    for y in years:

        yearly_fc = reduce_temperature_to_districts(y)

        merged = merged.merge(yearly_fc)

    return merged


# ------------------------------------------------------------
# 9) SCRIPT ENTRY POINT
# ------------------------------------------------------------
# When this script is run directly, it:
#
# 1. Builds the temperature feature table
# 2. Exports results to Google Drive
# ------------------------------------------------------------

if __name__ == "__main__":

    print("Running temperature feature extraction...")

    all_temp_fc = build_temperature_table_for_years(CONFIG["years"])

    selectors = [
        CONFIG["district_id_field"],
        "year",
        "GDD_03_06",
        "HeatDays_>27C_05_06",
        "Median_Tmax_Mar",
        "Median_Tmax_Apr",
        "Median_Tmax_May",
        "Median_Tmax_Jun",
        "Median_Tmin_Mar",
        "Median_Tmin_Apr",
        "Median_Tmin_May",
        "Median_Tmin_Jun",
    ]

    all_temp_fc = all_temp_fc.select(selectors)

    task = ee.batch.Export.table.toDrive(
        collection=all_temp_fc,
        description="temperature_features_2017_2023",
        folder=CONFIG["export_folder"],
        fileNamePrefix="temperature_features_2017_2023",
        fileFormat="CSV",
        selectors=selectors
    )

    task.start()

    print("Export started successfully.")
    print("Check the Earth Engine Tasks tab.")