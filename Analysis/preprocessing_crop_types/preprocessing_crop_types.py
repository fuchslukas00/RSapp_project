from pathlib import Path
import rasterio
from rasterio.merge import merge
import numpy as np

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

base_folder = Path(r"C:\Users\lukas\Documents\Studium\Remote_Sensing_Products\project\crop_yields\data\crop_types_D\crop_types\Results\tifs_by_year")
WHEAT_CLASS = 1110


year_folders = [f for f in base_folder.iterdir() if f.is_dir()]
print(f"Found {len(year_folders)} year folders.")

for year_folder in sorted(year_folders):
    year = year_folder.name
    print("\n-------------------------------------")
    print(f"Processing year: {year}")

    tif_files = list(year_folder.glob("*.tif"))
    if not tif_files:
        print("No TIF files found. Skipping.")
        continue

    print(f"Found {len(tif_files)} tiles.")

    # Temporary folder for reclassified tiles
    temp_dir = year_folder / "temp_binary_tiles"
    temp_dir.mkdir(exist_ok=True)

    binary_tile_paths = []

    # --------------------------------------------------------
    # 1) Reclassify each tile individually
    # --------------------------------------------------------
    for tif_path in tif_files:
        with rasterio.open(tif_path) as src:
            data = src.read(1)

            # Reclassify to binary uint8
            binary = (data == WHEAT_CLASS).astype("uint8")

            out_meta = src.meta.copy()
            out_meta.update({
                "dtype": "uint8",
                "count": 1,
                "compress": "lzw"
            })

            out_path = temp_dir / f"{tif_path.stem}_binary.tif"

            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(binary, 1)

            binary_tile_paths.append(out_path)

    print("Tile-wise reclassification done.")

    # --------------------------------------------------------
    # 2) Merge binary tiles
    # --------------------------------------------------------
    src_files = [rasterio.open(fp) for fp in binary_tile_paths]

    mosaic, out_transform = merge(src_files)
    mosaic = mosaic[0].astype("uint8")

    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[0],
        "width": mosaic.shape[1],
        "transform": out_transform,
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw"
    })

    output_file = base_folder / f"wheat_mask_{year}.tif"

    with rasterio.open(output_file, "w", **out_meta) as dest:
        dest.write(mosaic, 1)

    print(f"Saved wheat mask to: {output_file}")

    for src in src_files:
        src.close()

print("\nAll years processed.")