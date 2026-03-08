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
    # The district layer should contain at least:
    # - one unique ID field
    # - one readable name field
    # --------------------------------------------------------
    "districts_asset": "projects/rsapp-25-26/assets/germany_districts",

    # Name of the unique district ID field in the boundary layer.
    # Example: "NUTS_ID", "district_id", "LK_ID"
    "district_id_field": "district_id",

    # Name of the district name field.
    # Example: "NAME_LATN", "district_name"
    "district_name_field": "district_name",

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
# The district boundary layer is loaded as a FeatureCollection.
# We immediately keep only the essential fields:
# - district ID
# - district name
#
# This helps keep the object clean and avoids carrying many
# unnecessary attributes through the workflow.
# ------------------------------------------------------------

districts = ee.FeatureCollection(CONFIG["districts_asset"]).select(
    [CONFIG["district_id_field"], CONFIG["district_name_field"]]
)

# Geometry of the whole study area (union of all districts)
roi = districts.geometry()


# ------------------------------------------------------------
# 4) LOAD CROP MASK COLLECTION
# ------------------------------------------------------------
# At this stage we only load the ImageCollection object.
# We do NOT yet extract a specific year.
# That will be done in a later function.
# ------------------------------------------------------------

crop_mask_collection = ee.ImageCollection(CONFIG["crop_mask_asset"])


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

# Number of district features
district_count = districts.size().getInfo()
print(f"Number of district features: {district_count}")

# Number of crop mask images
crop_mask_count = crop_mask_collection.size().getInfo()
print(f"Number of crop mask images: {crop_mask_count}")

# Print the configured years
print("Configured years:", CONFIG["years"])

# Print the season windows
print("Season months:", CONFIG["season_months"])
print("Early season months:", CONFIG["early_months"])
print("Mid season months:", CONFIG["mid_months"])
print("Late season months:", CONFIG["late_months"])


# ------------------------------------------------------------
# 6) OPTIONAL: INSPECT THE FIRST DISTRICT
# ------------------------------------------------------------
# This helps confirm that the selected attribute names are
# correct and available in the uploaded district asset.
# ------------------------------------------------------------

first_district = districts.first().getInfo()
print("\nExample district feature:")
print(first_district)


# ------------------------------------------------------------
# 7) OPTIONAL: INSPECT THE FIRST CROP MASK IMAGE
# ------------------------------------------------------------
# This helps check whether:
# - the collection contains images
# - the "year" property exists
# - the expected crop mask band exists
# ------------------------------------------------------------

first_crop_mask = crop_mask_collection.first().getInfo()
print("\nExample crop mask image metadata:")
print(first_crop_mask)
