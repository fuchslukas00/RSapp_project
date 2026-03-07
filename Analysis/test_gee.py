import ee
import geemap

# ------------------------------------------------------------
# 1) Initialisierung
# ------------------------------------------------------------

# Falls noch nicht authentifiziert:
# ee.Authenticate()

ee.Initialize(project="rsapp-25-26")  # <-- HIER dein Projekt eintragen


# ------------------------------------------------------------
# 2) Deutschland als Region laden
# ------------------------------------------------------------

# FAO GAUL administrative boundaries
countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
germany = countries.filter(ee.Filter.eq("ADM0_NAME", "Germany"))

roi = germany.geometry()


# ------------------------------------------------------------
# 3) ERA5-Land Daily Aggregated laden (2023)
# ------------------------------------------------------------

era5 = (
    ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    .filterDate("2023-01-01", "2024-01-01")
    .select("total_precipitation_sum")
)

# ------------------------------------------------------------
# 4) Jahresniederschlag berechnen
# ------------------------------------------------------------

# ERA5 precipitation ist in Meter Wasseräquivalent
# -> *1000 für mm
precip_2023 = (
    era5
    .sum()
    .multiply(1000)
    .rename("precip_mm")
    .clip(roi)
)


# ------------------------------------------------------------
# 5) Karte anzeigen
# ------------------------------------------------------------

Map = geemap.Map(center=[51, 10], zoom=6)

vis_params = {
    "min": 400,
    "max": 1200,
    "palette": ["white", "blue", "darkblue"]
}

Map.addLayer(precip_2023, vis_params, "Mean Precip 2023 (mm)")
Map.addLayer(germany, {}, "Germany boundary")

Map

Map.save("map.html")
