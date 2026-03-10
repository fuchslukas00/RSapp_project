# gee_setup.py

import ee
from config import CONFIG


def init_gee():
    """Initialize the Earth Engine Python API."""
    ee.Initialize(project=CONFIG["gee_project"])


def load_districts_raw():
    """Load the full combined district + observed data asset."""
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


def get_observed_table(measure_key):
    """
    Return observed rows (yield or area) from the combined boundary asset.

    Args:
        measure_key: "yield" or "area"
    """
    if measure_key not in CONFIG["yield_measure_values"]:
        raise ValueError(
            f"Unknown measure_key '{measure_key}'. "
            f"Use one of: {list(CONFIG['yield_measure_values'].keys())}"
        )

    districts_raw = load_districts_raw()
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