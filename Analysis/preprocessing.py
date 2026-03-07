import ee

# ------------------------------------------------------------
# 0) EARTH ENGINE INITIALISIEREN
# ------------------------------------------------------------
# WICHTIG:
# Seit 2025 braucht die Python-Initialisierung in den üblichen
# Auth-Modi ein Cloud-Projekt.
# Beispiel:
# ee.Authenticate()
# ee.Initialize(project="dein-gcp-projekt")
#
# Wenn du bereits authentifiziert bist:
ee.Initialize(project="rsapp-25-26")


# ------------------------------------------------------------
# 1) KONFIGURATION
# ------------------------------------------------------------

CONFIG = {
    # ---- Eingabedaten / Assets ----
    # District-Grenzen (dein eigenes Asset oder ein importiertes Shapefile)
    "districts_asset": "users/DEIN_USERNAME/germany_districts",
    "district_id_field": "district_id",      # eindeutige ID
    "district_name_field": "district_name",  # lesbarer Name

    # Crop mask als ImageCollection:
    # Erwartung:
    # - jedes Bild hat property "year"
    # - Bandname unten in crop_mask_band
    # - Zielkultur = crop_mask_value (z.B. 1)
    "crop_mask_asset": "users/DEIN_USERNAME/cropmask_targetcrop_by_year",
    "crop_mask_band": "crop_mask",
    "crop_mask_value": 1,

    # ---- Zeitraum ----
    "years": list(range(2017, 2024)),

    # ---- Saisondefinition ----
    # Diese Monatsfenster musst du fachlich begründen.
    # Beispiel hier nur als Template.
    "season_months": [3, 4, 5, 6, 7],   # Gesamtsaison für NDVI-Integral etc.
    "early_months":  [3, 4],
    "mid_months":    [5, 6],
    "late_months":   [7],

    # ---- Temperatur / Niederschlag Parameter ----
    "gdd_base_temp_c": 5.0,     # kulturabhängig begründen
    "hot_day_threshold_c": 30.0,
    "rainy_day_threshold_mm": 1.0,

    # ---- Bodenfeuchte-Anomalie ----
    "soil_moisture_baseline_years": list(range(2017, 2024)),

    # ---- Sentinel-2 Cloud Filtering ----
    "s2_cloud_probability_threshold": 40,

    # ---- Skalen für reduceRegions ----
    # Sentinel-2 nominal ~10m
    "s2_scale": 10,
    # ERA5-Land ~11 km
    "era5_scale": 11132,

    # ---- Export ----
    "export_description": "district_predictors_export",
    "export_folder": "GEE_exports",
    "export_file_prefix": "district_predictors"
}


# ------------------------------------------------------------
# 2) BASISDATEN LADEN
# ------------------------------------------------------------

districts = ee.FeatureCollection(CONFIG["districts_asset"]).select(
    [CONFIG["district_id_field"], CONFIG["district_name_field"]]
)

roi = districts.geometry()


# ------------------------------------------------------------
# 3) HILFSFUNKTIONEN
# ------------------------------------------------------------

def month_list(months):
    """Python-Liste -> ee.List"""
    return ee.List(months)


def start_of_month(year, month):
    return ee.Date.fromYMD(year, month, 1)


def end_of_month(year, month):
    return ee.Date.fromYMD(year, month, 1).advance(1, "month")


def season_start_end(year, months):
    """
    Liefert (start_date, end_date) für eine Monatsliste wie [3,4,5,6,7].
    end_date ist exklusiv.
    """
    months_sorted = sorted(months)
    start = ee.Date.fromYMD(year, months_sorted[0], 1)
    end = ee.Date.fromYMD(year, months_sorted[-1], 1).advance(1, "month")
    return start, end


def get_crop_mask(year):
    """
    Holt die Crop-Mask für ein bestimmtes Jahr.
    Erwartet:
    - ImageCollection mit property 'year'
    - Bandname CONFIG['crop_mask_band']
    - Zielkultur = CONFIG['crop_mask_value']
    """
    mask_img = (
        ee.ImageCollection(CONFIG["crop_mask_asset"])
        .filter(ee.Filter.eq("year", year))
        .first()
    )

    # Binäre Maske: 1 = Zielkultur, sonst maskiert
    crop_mask = (
        ee.Image(mask_img)
        .select(CONFIG["crop_mask_band"])
        .eq(CONFIG["crop_mask_value"])
        .selfMask()
        .clip(roi)
    )

    return crop_mask


def add_s2_cloud_mask_and_ndvi(img):
    """
    Nutzt das verknüpfte Cloud-Probability-Bild und berechnet NDVI.
    """
    cloud_img = ee.Image(img.get("cloud_mask"))
    clear = cloud_img.select("probability").lt(
        CONFIG["s2_cloud_probability_threshold"]
    )

    ndvi = (
        img.updateMask(clear)
        .normalizedDifference(["B8", "B4"])
        .rename("NDVI")
        .copyProperties(img, ["system:time_start"])
    )

    return ndvi


def get_s2_ndvi_collection(year):
    """
    Sentinel-2 SR + Cloud Probability joinen, dann NDVI berechnen.
    """
    start, end = season_start_end(year, CONFIG["season_months"])

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(roi)
        .filterDate(start, end)
    )

    s2_clouds = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(roi)
        .filterDate(start, end)
    )

    # Join über system:index
    joined = ee.Join.saveFirst("cloud_mask").apply(
        primary=s2,
        secondary=s2_clouds,
        condition=ee.Filter.equals(
            leftField="system:index",
            rightField="system:index"
        )
    )

    ndvi_ic = ee.ImageCollection(joined).map(add_s2_cloud_mask_and_ndvi)
    return ndvi_ic


def get_monthly_ndvi_images(year, crop_mask):
    """
    Baut für jeden Monat der season_months ein Monats-Median-NDVI-Bild.
    Diese Monatsbilder sind wichtig, weil das NDVI-Integral sonst von der
    Anzahl verfügbarer Szenen abhängt.
    """
    months = CONFIG["season_months"]

    monthly_images = []
    for m in months:
        start = start_of_month(year, m)
        end = end_of_month(year, m)

        monthly_ndvi = (
            get_s2_ndvi_collection(year)
            .filterDate(start, end)
            .median()
            .updateMask(crop_mask)
            .rename("NDVI")
            .set("month", m)
            .set("system:time_start", start.millis())
        )

        # Zusätzlicher Band mit Monatsnummer.
        # So kann qualityMosaic('NDVI') später auch den "peak month" mitziehen.
        month_band = ee.Image.constant(m).rename("peak_month").toFloat()
        monthly_ndvi = monthly_ndvi.addBands(month_band)

        monthly_images.append(monthly_ndvi)

    return ee.ImageCollection.fromImages(monthly_images)


def mean_ndvi_for_months(monthly_ndvi_ic, months_subset):
    """
    Mittleres NDVI über ein Monatsfenster (z.B. early oder mid).
    """
    months_subset = ee.List(months_subset)

    return (
        monthly_ndvi_ic
        .filter(ee.Filter.inList("month", months_subset))
        .select("NDVI")
        .mean()
    )


def ndvi_peak_and_peak_month(monthly_ndvi_ic):
    """
    Nimmt pro Pixel das Monatsbild mit maximalem NDVI.
    Ergebnis:
    - peak NDVI
    - peak month
    WICHTIG:
    Das ist pixelbasiert; nach reduceRegions wird der District-Mittelwert
    des Pixel-Peak-Monats exportiert.
    """
    peak_img = monthly_ndvi_ic.qualityMosaic("NDVI")
    peak_ndvi = peak_img.select("NDVI").rename("ndvi_peak")
    peak_month = peak_img.select("peak_month").rename("ndvi_peak_month")
    return peak_ndvi, peak_month


def ndvi_integral(monthly_ndvi_ic):
    """
    Einfaches saisonales NDVI-Integral als Summe der Monats-Median-NDVI-Bilder.
    Für eine erste Studie ein robuster, leicht begründbarer Proxy.
    """
    return monthly_ndvi_ic.select("NDVI").sum().rename("ndvi_integral")


def get_era5_daily(year):
    """
    ERA5-Land Daily Aggregated für ein Jahr.
    """
    start, end = season_start_end(year, CONFIG["season_months"])

    era5 = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterBounds(roi)
        .filterDate(start, end)
    )

    return era5


def mean_temp_window(year, months):
    """
    Mittlere 2m-Temperatur (°C) über ein Monatsfenster.
    ERA5-Land Temperatur ist in Kelvin.
    """
    start, end = season_start_end(year, months)

    img = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start, end)
        .select("temperature_2m")
        .mean()
        .subtract(273.15)
        .rename(f"temp_mean_{months[0]}_{months[-1]}")
    )

    return img


def gdd_sum(year, months):
    """
    Growing Degree Days:
    Summe über max(Tmean - base_temp, 0) pro Tag.
    """
    base_temp = CONFIG["gdd_base_temp_c"]
    start, end = season_start_end(year, months)

    def daily_gdd(img):
        t_c = img.select("temperature_2m").subtract(273.15)
        gdd = t_c.subtract(base_temp).max(0).rename("gdd")
        return gdd.copyProperties(img, ["system:time_start"])

    gdd_img = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start, end)
        .map(daily_gdd)
        .sum()
        .rename(f"gdd_sum_{months[0]}_{months[-1]}")
    )

    return gdd_img


def hot_days_count(year, months):
    """
    Anzahl Tage mit Tmax > Schwelle.
    ERA5-Land hat temperature_2m_max.
    """
    threshold_k = CONFIG["hot_day_threshold_c"] + 273.15
    start, end = season_start_end(year, months)

    def hot_day(img):
        hot = img.select("temperature_2m_max").gt(threshold_k).rename("hot_day")
        return hot.copyProperties(img, ["system:time_start"])

    hot_img = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start, end)
        .map(hot_day)
        .sum()
        .rename(f"hot_days_{months[0]}_{months[-1]}")
    )

    return hot_img


def precip_sum(year, months):
    """
    Niederschlagssumme (mm) über ein Monatsfenster.
    ERA5-Land total_precipitation_sum ist in Metern Wasseräquivalent.
    -> *1000 für mm
    -> clamp(0) gegen kleine negative Artefakte
    """
    start, end = season_start_end(year, months)

    def daily_p(img):
        p_mm = (
            img.select("total_precipitation_sum")
            .max(0)               # negative Artefakte vermeiden
            .multiply(1000.0)     # m -> mm
            .rename("precip_mm")
        )
        return p_mm.copyProperties(img, ["system:time_start"])

    p_img = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start, end)
        .map(daily_p)
        .sum()
        .rename(f"precip_sum_{months[0]}_{months[-1]}")
    )

    return p_img


def rainy_days_count(year, months):
    """
    Anzahl Tage mit Tagesniederschlag >= rainy_day_threshold_mm
    """
    threshold_mm = CONFIG["rainy_day_threshold_mm"]
    start, end = season_start_end(year, months)

    def rainy_day(img):
        p_mm = img.select("total_precipitation_sum").max(0).multiply(1000.0)
        rainy = p_mm.gte(threshold_mm).rename("rainy_day")
        return rainy.copyProperties(img, ["system:time_start"])

    rainy_img = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start, end)
        .map(rainy_day)
        .sum()
        .rename(f"rainy_days_{months[0]}_{months[-1]}")
    )

    return rainy_img


def soil_moisture_anomaly(year, months):
    """
    Bodenfeuchte-Anomalie = Jahreswert im Fenster minus Baseline-Mittel
    Band: volumetric_soil_water_layer_1 (oberste Schicht)
    Für District-Level ist das ein sinnvoller erster Ansatz.

    WICHTIG:
    Den Baseline-Zeitraum musst du im Paper begründen.
    """
    # Jahreswert
    start, end = season_start_end(year, months)
    year_sm = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .filterDate(start, end)
        .select("volumetric_soil_water_layer_1")
        .mean()
    )

    # Baseline über mehrere Jahre, aber dieselben Monate
    baseline_imgs = []
    for y in CONFIG["soil_moisture_baseline_years"]:
        bs_start, bs_end = season_start_end(y, months)
        baseline_y = (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterDate(bs_start, bs_end)
            .select("volumetric_soil_water_layer_1")
            .mean()
        )
        baseline_imgs.append(baseline_y)

    baseline_mean = ee.ImageCollection.fromImages(baseline_imgs).mean()

    anomaly = (
        year_sm.subtract(baseline_mean)
        .rename(f"soil_moisture_anom_{months[0]}_{months[-1]}")
    )

    return anomaly


def build_predictor_stack_for_year(year):
    """
    Baut für ein Jahr einen Multiband-Image-Stack aller Predictor-Variablen.
    """
    year = int(year)

    # -------------------------
    # Crop mask
    # -------------------------
    crop_mask = get_crop_mask(year)

    # -------------------------
    # NDVI
    # -------------------------
    monthly_ndvi_ic = get_monthly_ndvi_images(year, crop_mask)

    ndvi_early = mean_ndvi_for_months(monthly_ndvi_ic, CONFIG["early_months"]).rename("ndvi_mean_early")
    ndvi_mid = mean_ndvi_for_months(monthly_ndvi_ic, CONFIG["mid_months"]).rename("ndvi_mean_mid")
    ndvi_late = mean_ndvi_for_months(monthly_ndvi_ic, CONFIG["late_months"]).rename("ndvi_mean_late")

    ndvi_peak, ndvi_peak_month = ndvi_peak_and_peak_month(monthly_ndvi_ic)
    ndvi_integral_img = ndvi_integral(monthly_ndvi_ic)

    # -------------------------
    # Klima
    # -------------------------
    temp_early = mean_temp_window(year, CONFIG["early_months"]).rename("temp_mean_early")
    temp_mid = mean_temp_window(year, CONFIG["mid_months"]).rename("temp_mean_mid")
    temp_late = mean_temp_window(year, CONFIG["late_months"]).rename("temp_mean_late")

    gdd_img = gdd_sum(year, CONFIG["season_months"]).rename("gdd_sum_season")
    hot_days_img = hot_days_count(year, CONFIG["mid_months"]).rename("hot_days_mid")

    precip_early = precip_sum(year, CONFIG["early_months"]).rename("precip_sum_early")
    precip_mid = precip_sum(year, CONFIG["mid_months"]).rename("precip_sum_mid")
    precip_late = precip_sum(year, CONFIG["late_months"]).rename("precip_sum_late")

    rainy_days_img = rainy_days_count(year, CONFIG["season_months"]).rename("rainy_days_season")

    # -------------------------
    # Bodenfeuchte-Anomalie
    # -------------------------
    sm_anom_img = soil_moisture_anomaly(year, CONFIG["mid_months"]).rename("soil_moisture_anom_mid")

    # -------------------------
    # Multiband-Stack
    # -------------------------
    stack = ee.Image.cat([
        ndvi_early,
        ndvi_mid,
        ndvi_late,
        ndvi_peak,
        ndvi_peak_month,
        ndvi_integral_img,
        temp_early,
        temp_mid,
        temp_late,
        gdd_img,
        hot_days_img,
        precip_early,
        precip_mid,
        precip_late,
        rainy_days_img,
        sm_anom_img
    ])

    return stack


def reduce_stack_to_districts(year):
    """
    Reduziert den Predictor-Stack auf District-Ebene.
    Ergebnis: FeatureCollection mit einer Zeile pro District.
    """
    stack = build_predictor_stack_for_year(year)

    # Ein einziger mean-Reducer für alle Bänder.
    # Für dein Setup (District x Jahr) ist das die sauberste erste Variante.
    reduced = stack.reduceRegions(
        collection=districts,
        reducer=ee.Reducer.mean(),
        scale=CONFIG["s2_scale"]  # Für den Stack nimmt GEE intern je Band die passende Scale.
    )

    # Jahresinformation an jede Zeile hängen
    reduced = reduced.map(
        lambda f: f.set("year", year)
    )

    return reduced


# ------------------------------------------------------------
# 4) ALLE JAHRE BERECHNEN UND ZUSAMMENFÜHREN
# ------------------------------------------------------------

all_fc = ee.FeatureCollection([])

for y in CONFIG["years"]:
    yearly_fc = reduce_stack_to_districts(y)
    all_fc = all_fc.merge(yearly_fc)


# ------------------------------------------------------------
# 5) OPTIONAL: SPALTENREIHENFOLGE BEREINIGEN
# ------------------------------------------------------------
# Nur die wichtigsten Felder behalten.
selectors = [
    CONFIG["district_id_field"],
    CONFIG["district_name_field"],
    "year",
    "ndvi_mean_early",
    "ndvi_mean_mid",
    "ndvi_mean_late",
    "ndvi_peak",
    "ndvi_peak_month",
    "ndvi_integral",
    "temp_mean_early",
    "temp_mean_mid",
    "temp_mean_late",
    "gdd_sum_season",
    "hot_days_mid",
    "precip_sum_early",
    "precip_sum_mid",
    "precip_sum_late",
    "rainy_days_season",
    "soil_moisture_anom_mid"
]

all_fc = all_fc.select(selectors)


# ------------------------------------------------------------
# 6) EXPORT ALS CSV NACH GOOGLE DRIVE
# ------------------------------------------------------------

task = ee.batch.Export.table.toDrive(
    collection=all_fc,
    description=CONFIG["export_description"],
    folder=CONFIG["export_folder"],
    fileNamePrefix=CONFIG["export_file_prefix"],
    fileFormat="CSV",
    selectors=selectors
)

task.start()

print("Export gestartet. Status später mit task.status() prüfen.")
