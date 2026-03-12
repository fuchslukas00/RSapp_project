# config.py

CONFIG = {
    # --------------------------------------------------------
    # Google Earth Engine project
    # --------------------------------------------------------
    "gee_project": "rsapp-25-26",

    # --------------------------------------------------------
    # Spatial boundary dataset: NUTS3 districts for Germany
    # --------------------------------------------------------
    "districts_asset": "projects/rsapp-25-26/assets/nuts3_ger",

    # Unique district ID field in the new boundary asset
    "district_id_field": "NUTS_CODE",

    # Optional readable district name field
    "district_name_field": None,

    # --------------------------------------------------------
    # Optional observed yield / area table asset
    # --------------------------------------------------------
    # Keep this flexible in case observed rows are stored in a
    # separate asset or merged later outside GEE.
    # If None, gee_setup.get_observed_table() will raise a clear
    # error instead of assuming the boundary asset contains data.
    "observed_table_asset": None,
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
    # - year encoded in system:index or stored as property
    # - band storing binary crop mask
    "crop_mask_asset": "projects/rsapp-25-26/assets/cropmask_targetcrop_by_year_ger",
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

    # --------------------------------------------------------
    # Precipitation feature windows
    # --------------------------------------------------------
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
    "export_suffix": "ger_2017_2023",
}
