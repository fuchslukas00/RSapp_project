# ============================================================
# NDVI FEATURES FROM SENTINEL-2
# ============================================================
# This script computes NDVI-based predictors for crop yield
# modelling at district level.
#
# Data sources:
# - Sentinel-2 Surface Reflectance Harmonized
# - Sentinel-2 Cloud Probability
# - Annual crop mask image collection
#
# Workflow:
# 1. Initialize Google Earth Engine
# 2. Load district geometries
# 3. Load yearly wheat crop mask
# 4. Build monthly NDVI composites (March-June)
# 5. Derive seasonal NDVI features
# 6. Aggregate features to district level
# 7. Export results as CSV
#
# Output:
# One row per district-year with NDVI predictors.
# ============================================================

import ee

from config import CONFIG
from gee_setup import init_gee, load_districts, get_roi


# ------------------------------------------------------------
# 1) INITIALIZE EARTH ENGINE
# ------------------------------------------------------------

init_gee()


# ------------------------------------------------------------
# 2) LOAD DISTRICT GEOMETRIES AND ROI
# ------------------------------------------------------------
# districts:
# one geometry feature per district
#
# roi:
# union of all district geometries, used to spatially filter
# Sentinel-2 collections
# ------------------------------------------------------------

districts = load_districts()
roi = get_roi()


# ------------------------------------------------------------
# 3) DEFINE INPUT DATASETS
# ------------------------------------------------------------
# Sentinel-2 SR Harmonized:
# surface reflectance data used to calculate NDVI
#
# Sentinel-2 Cloud Probability:
# cloud probability product used for cloud masking
#
# Crop mask:
# annual ImageCollection uploaded by you
# expected values:
# - 1 = target crop (wheat)
# - 0 = not target crop
# ------------------------------------------------------------

S2_SR_ID = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_ID = "COPERNICUS/S2_CLOUD_PROBABILITY"

crop_mask_collection = ee.ImageCollection(CONFIG["crop_mask_asset"])


# ------------------------------------------------------------
# 4) NDVI CONFIGURATION
# ------------------------------------------------------------
# We explicitly define the monthly NDVI window here.
# For winter wheat in your current setup:
# March-June
#
# If you want this fully centralized later, you can move it
# into config.py as:
# "monthly_ndvi_months": [3, 4, 5, 6]
# ------------------------------------------------------------

NDVI_MONTHS = [3, 4, 5, 6]

MONTH_NAME_MAP = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}


# ------------------------------------------------------------
# 5) HELPER FUNCTION: MONTH WINDOW
# ------------------------------------------------------------
# Returns [start, end) date window for one month.
#
# Example:
# _month_window(2020, 3)
# -> 2020-03-01 to 2020-04-01
# ------------------------------------------------------------

def _month_window(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    return start, end


# ------------------------------------------------------------
# 6) LOAD YEARLY CROP MASK
# ------------------------------------------------------------
# This function selects the crop mask image for one year.
#
# Assumption:
# each image in the ImageCollection has a property "year"
#
# If your uploaded assets do NOT have a property "year",
# then this function must be adapted.
# ------------------------------------------------------------

def get_crop_mask(year):
    """
    Select crop mask image for a given year based on
    the year encoded in the asset name (system:index).
    """

    mask_img = (
        crop_mask_collection
        .filter(ee.Filter.stringContains("system:index", str(year)))
        .first()
    )

    crop_mask = (
        ee.Image(mask_img)
        .select(CONFIG["crop_mask_band"])
        .eq(CONFIG["crop_mask_value"])
        .selfMask()
        .clip(roi)
    )

    return crop_mask


# ------------------------------------------------------------
# 7) BUILD SENTINEL-2 NDVI COLLECTION
# ------------------------------------------------------------
# Steps:
# 1. Filter Sentinel-2 SR and cloud probability collections
# 2. Join them using system:index
# 3. Apply cloud mask
# 4. Compute NDVI
#
# NDVI = (NIR - Red) / (NIR + Red)
# Sentinel-2 bands:
# - B8 = NIR
# - B4 = Red
# ------------------------------------------------------------

def _add_s2_cloud_mask_and_ndvi(img):
    """
    Use the joined cloud probability image to mask cloudy pixels
    and compute NDVI.
    """
    cloud_img = ee.Image(img.get("cloud_mask"))

    clear_mask = cloud_img.select("probability").lt(
        CONFIG["s2_cloud_probability_threshold"]
    )

    ndvi = (
        img.updateMask(clear_mask)
        .normalizedDifference(["B8", "B4"])
        .rename("NDVI")
        .copyProperties(img, ["system:time_start"])
    )

    return ndvi


def get_s2_ndvi_collection(year):
    """
    Load Sentinel-2 NDVI collection for the configured NDVI months.
    """
    start = ee.Date.fromYMD(year, min(NDVI_MONTHS), 1)
    end = ee.Date.fromYMD(year, max(NDVI_MONTHS), 1).advance(1, "month")

    s2 = (
        ee.ImageCollection(S2_SR_ID)
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 80))
    )

    s2_clouds = (
        ee.ImageCollection(S2_CLOUD_ID)
        .filterBounds(roi)
        .filterDate(start, end)
    )

    # Join Sentinel-2 SR and cloud probability images by system:index
    joined = ee.Join.saveFirst("cloud_mask").apply(
        primary=s2,
        secondary=s2_clouds,
        condition=ee.Filter.equals(
            leftField="system:index",
            rightField="system:index"
        )
    )

    ndvi_ic = ee.ImageCollection(joined).map(_add_s2_cloud_mask_and_ndvi)

    return ndvi_ic


# ------------------------------------------------------------
# 8) BUILD MONTHLY NDVI IMAGES
# ------------------------------------------------------------
# For each month (March-June), build one monthly NDVI composite.
#
# We use monthly median NDVI because it is robust to:
# - remaining cloud contamination
# - outliers
# - noisy observations
#
# The crop mask is applied here, so all later NDVI features
# are computed only over wheat pixels.
# ------------------------------------------------------------

def get_monthly_ndvi_images(year, crop_mask):
    """
    Build one monthly NDVI image for each month in NDVI_MONTHS.
    """
    monthly_images = []

    full_ndvi_ic = get_s2_ndvi_collection(year)

    for month in NDVI_MONTHS:
        month_start, month_end = _month_window(year, month)

        monthly_ndvi = (
            full_ndvi_ic
            .filterDate(month_start, month_end)
            .median()
            .updateMask(crop_mask)
            .rename("NDVI")
            .set("month", month)
            .set("system:time_start", month_start.millis())
        )

        # Add a month band so we can later retrieve peak timing
        month_band = ee.Image.constant(month).rename("peak_month").toFloat()
        monthly_ndvi = monthly_ndvi.addBands(month_band)

        monthly_images.append(monthly_ndvi)

    return ee.ImageCollection.fromImages(monthly_images)


# ------------------------------------------------------------
# 9) BUILD NDVI FEATURE IMAGE
# ------------------------------------------------------------
# Features:
# - monthly NDVI for Mar, Apr, May, Jun
# - NDVI_peak
# - NDVI_peak_month
# - NDVI_integral
#
# Definitions:
# - monthly NDVI = monthly median NDVI
# - NDVI_peak = maximum monthly NDVI
# - NDVI_peak_month = month in which the pixel-level peak occurs
# - NDVI_integral = sum of monthly NDVI values (Mar-Jun)
# ------------------------------------------------------------

def build_ndvi_feature_image(year):
    """
    Build one multi-band image with all NDVI features for one year.
    """
    crop_mask = get_crop_mask(year)

    monthly_ndvi_ic = get_monthly_ndvi_images(year, crop_mask)

    # --------------------------------------------------------
    # 9.1 Monthly NDVI bands
    # --------------------------------------------------------
    monthly_bands = []

    for month in NDVI_MONTHS:
        month_label = MONTH_NAME_MAP[month]

        month_img = (
            monthly_ndvi_ic
            .filter(ee.Filter.eq("month", month))
            .first()
            .select("NDVI")
            .rename(f"NDVI_{month_label}")
        )

        monthly_bands.append(month_img)

    # --------------------------------------------------------
    # 9.2 Peak NDVI and peak timing
    # --------------------------------------------------------
    # qualityMosaic("NDVI") keeps, for each pixel, the image
    # where NDVI is highest.
    # From that image we extract:
    # - peak NDVI value
    # - peak month
    # --------------------------------------------------------
    peak_img = monthly_ndvi_ic.qualityMosaic("NDVI")

    ndvi_peak = peak_img.select("NDVI").rename("NDVI_peak")
    ndvi_peak_month = peak_img.select("peak_month").rename("NDVI_peak_month")

    # --------------------------------------------------------
    # 9.3 NDVI integral
    # --------------------------------------------------------
    # Simple seasonal NDVI integral:
    # sum of monthly median NDVI values
    #
    # This acts as a proxy for cumulative seasonal greenness.
    # --------------------------------------------------------
    ndvi_integral = (
        monthly_ndvi_ic
        .select("NDVI")
        .sum()
        .rename("NDVI_integral")
    )

    ndvi_stack = ee.Image.cat(
        monthly_bands + [ndvi_peak, ndvi_peak_month, ndvi_integral]
    ).clip(roi)

    return ndvi_stack


# ------------------------------------------------------------
# 10) AGGREGATE NDVI FEATURES TO DISTRICT LEVEL
# ------------------------------------------------------------
# Pixel-level NDVI features are aggregated to district level
# using reduceRegions() and the mean reducer.
#
# This gives one row per district per year.
# ------------------------------------------------------------

def reduce_ndvi_to_districts(year):
    """
    Aggregate NDVI features to district polygons.
    """
    ndvi_stack = build_ndvi_feature_image(year)

    reduced = ndvi_stack.reduceRegions(
        collection=districts,
        reducer=ee.Reducer.mean(),
        scale=CONFIG["s2_scale"],
    )

    return reduced.map(lambda f: f.set("year", year))


# ------------------------------------------------------------
# 11) BUILD FULL NDVI TABLE FOR ALL YEARS
# ------------------------------------------------------------
# Loops over all configured years and merges district-year
# results into one FeatureCollection.
# ------------------------------------------------------------

def build_ndvi_table_for_years(years):
    """
    Return one FeatureCollection with all district-year NDVI features.
    """
    merged = ee.FeatureCollection([])

    for y in years:
        yearly_fc = reduce_ndvi_to_districts(y)
        merged = merged.merge(yearly_fc)

    return merged


# ------------------------------------------------------------
# 12) SCRIPT ENTRY POINT
# ------------------------------------------------------------
# When this script is run directly, it:
# 1. Performs a quick one-year check
# 2. Builds the NDVI feature table for all years
# 3. Exports results to Google Drive
# ------------------------------------------------------------

if __name__ == "__main__":

    print("Running NDVI feature extraction...")

    # Quick check for one example year
    sample_year = CONFIG["years"][0]
    sample_ndvi_image = build_ndvi_feature_image(sample_year)

    #print("NDVI feature bands:")
    #print(sample_ndvi_image.bandNames().getInfo())

    sample_district_table = reduce_ndvi_to_districts(sample_year)
    #print(f"District rows for {sample_year}: {sample_district_table.size().getInfo()}")

    # Explicit export columns
    selectors = [
        CONFIG["district_id_field"],
        "year",
        "NDVI_Mar",
        "NDVI_Apr",
        "NDVI_May",
        "NDVI_Jun",
        "NDVI_peak",
        "NDVI_peak_month",
        "NDVI_integral",
    ]

    all_ndvi_fc = build_ndvi_table_for_years(CONFIG["years"])
    all_ndvi_fc = all_ndvi_fc.select(selectors)

    task = ee.batch.Export.table.toDrive(
        collection=all_ndvi_fc,
        description="ndvi_features_2017_2023",
        folder=CONFIG["export_folder"],
        fileNamePrefix="ndvi_features_2017_2023",
        fileFormat="CSV",
        selectors=selectors
    )

    task.start()

    print("✅ Export started successfully.")
    print("Check the Earth Engine Tasks tab or Google Drive.")