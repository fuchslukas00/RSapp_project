from pathlib import Path
import rasterio
from rasterio.merge import merge
import numpy as np

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

# Folder containing all 2022 tiles
input_folder = Path(r"C:\Users\lukas\Documents\Studium\Remote_Sensing_Products\project\crop_yields\data\crop_types\crop_types2\Results\tifs_by_year\2017")

# Output file
output_file = input_folder.parent / "wheat_mask_2017_bw.tif"

# Wheat class value in HRL
WHEAT_CLASS = 1110


# ------------------------------------------------------------
# 1) Load all raster tiles
# ------------------------------------------------------------

tif_files = list(input_folder.glob("*.tif"))

if not tif_files:
    raise ValueError("No TIF files found.")

print(f"Found {len(tif_files)} tiles.")

src_files = [rasterio.open(fp) for fp in tif_files]


# ------------------------------------------------------------
# 2) Mosaic all tiles
# ------------------------------------------------------------

mosaic, out_transform = merge(src_files)

# mosaic shape = (bands, height, width)
# HRL has 1 band
mosaic = mosaic[0]  # take first band

print("Mosaic created.")


# ------------------------------------------------------------
# 3) Reclassify: Wheat → 1, others → 0
# ------------------------------------------------------------

wheat_mask = np.where(mosaic == WHEAT_CLASS, 1, 0).astype("uint8")

print("Reclassification done.")


# ------------------------------------------------------------
# 4) Save result
# ------------------------------------------------------------

out_meta = src_files[0].meta.copy()

out_meta.update({
    "driver": "GTiff",
    "height": wheat_mask.shape[0],
    "width": wheat_mask.shape[1],
    "transform": out_transform,
    "count": 1,
    "dtype": "uint8",
    "compress": "lzw"
})

with rasterio.open(output_file, "w", **out_meta) as dest:
    dest.write(wheat_mask, 1)

print(f"Saved wheat mask to: {output_file}")


# ------------------------------------------------------------
# 5) Cleanup
# ------------------------------------------------------------

for src in src_files:
    src.close()

print("Done.")
