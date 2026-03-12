# gee_setup.py

import ee
from config import CONFIG


def init_gee():
    """Initialize the Earth Engine Python API."""
    ee.Initialize(project=CONFIG["gee_project"])


def load_districts_raw():
    """Load the district boundary asset."""
    return ee.FeatureCollection(CONFIG["districts_asset"])


def load_districts():
    """
    Load one geometry feature per district.
    This is the version used for spatial aggregation.
    """
    districts_raw = load_districts_raw()

    district_selectors = [CONFIG["district_id_field"]]
    if CONFIG["district_name_field"]:
        district_selectors.append(CONFIG["district_name_field"])

    districts = (
        districts_raw
        .select(district_selectors)
        .distinct([CONFIG["district_id_field"]])
    )
    return districts


def get_roi():
    """Return the union geometry of all districts."""
    return load_districts().geometry()


def load_observed_table_raw():
    """
    Load the observed yield / area table asset if configured.
    """
    observed_asset = CONFIG.get("observed_table_asset")
    if not observed_asset:
        raise ValueError(
            "CONFIG['observed_table_asset'] is not set. "
            "Your current Germany boundary asset only provides geometry. "
            "If you want to load observed yield/area rows in GEE, point this "
            "setting to the corresponding table asset."
        )
    return ee.FeatureCollection(observed_asset)


def get_observed_table(measure_key):
    """
    Return observed rows (yield or area) from the configured
    observed table asset.

    Args:
        measure_key: "yield" or "area"
    """
    if measure_key not in CONFIG["yield_measure_values"]:
        raise ValueError(
            f"Unknown measure_key '{measure_key}'. "
            f"Use one of: {list(CONFIG['yield_measure_values'].keys())}"
        )

    observed_raw = load_observed_table_raw()
    measure_value = CONFIG["yield_measure_values"][measure_key]

    return observed_raw.filter(
        ee.Filter.eq(CONFIG["yield_measure_field"], measure_value)
    ).select(
        [
            CONFIG["district_id_field"],
            CONFIG["yield_measure_field"],
            CONFIG["yield_year_field"],
            CONFIG["yield_value_field"],
        ]
    )
