# config.py

CONFIG = {
    # --------------------------------------------------------
    # Google Earth Engine project
    # --------------------------------------------------------
    "gee_project": "rsapp-25-26",

    # --------------------------------------------------------
    # Spatial boundary dataset: districts / NUTS3 / counties
    # --------------------------------------------------------
    # This asset contains:
    # - district geometry
    # - observed yield / area rows by district-year
    "districts_asset": "projects/rsapp-25-26/assets/district_boundaries",

    # Unique district ID field
    "district_id_field": "nuts_id",

    # Optional readable district name field
    "district_name_field": None,

    # --------------------------------------------------------
    # Observed yield / area fields in the same asset
    # --------------------------------------------------------
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
    # Expected structure:
    # - ImageCollection
    # - one image per year
    # - property "year"
    # - band storing binary crop mask
    "crop_mask_asset": "projects/rsapp-25-26/assets/cropmask_targetcrop_by_year",
    "crop_mask_band": "b1",
    "crop_mask_value": 1,

    # --------------------------------------------------------
    # Temporal setup
    # --------------------------------------------------------
    "years": list(range(2017, 2024)),

    # --------------------------------------------------------
    # Temperature feature windows
    # --------------------------------------------------------
    "gdd_months": [3, 4, 5, 6],
    "heat_months": [5, 6],
    "monthly_temp_months": [3, 4, 5, 6],

    #--------------------------------------------------------
    # Precipitation feature windows
    #--------------------------------------------------------
    "monthly_precip_months": [3, 4, 5, 6],

    # --------------------------------------------------------
    # Climate thresholds
    # --------------------------------------------------------
    "gdd_base_temp_c": 5.0,
    "hot_day_threshold_c": 27.0,
    "rainy_day_threshold_mm": 1.0,

    # --------------------------------------------------------
    # Soil moisture anomaly baseline years
    # --------------------------------------------------------
    "soil_moisture_baseline_years": list(range(2017, 2024)),

    # --------------------------------------------------------
    # Sentinel-2 cloud masking
    # --------------------------------------------------------
    "s2_cloud_probability_threshold": 40,

    # --------------------------------------------------------
    # Spatial scales
    # --------------------------------------------------------
    "s2_scale": 10,
    "era5_scale": 11132,

    # --------------------------------------------------------
    # Export settings
    # --------------------------------------------------------
    "export_folder": "GEE_exports",
}