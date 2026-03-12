from pathlib import Path
import zipfile
import re

# Root folder containing the ZIP files
src_dir = Path(r"C:\Users\lukas\Documents\Studium\Remote_Sensing_Products\project\crop_yields\data\crop_types_D\crop_types_2\Results")

# Output base folder
out_dir = src_dir / "extracted_by_year"
out_dir.mkdir(exist_ok=True)

# Regex to extract year from names like:
# CLMS_HRLVLCC_CTY_S2022_R10m_E41N27_...
year_pattern = re.compile(r"_S(20\d{2})_")

zip_files = list(src_dir.glob("*.zip"))

if not zip_files:
    print("No ZIP files found.")
else:
    print(f"Found {len(zip_files)} ZIP files.")

for zip_path in zip_files:
    match = year_pattern.search(zip_path.name)
    if not match:
        print(f"Skipping (no year found): {zip_path.name}")
        continue

    year = match.group(1)
    year_dir = out_dir / year
    year_dir.mkdir(exist_ok=True)

    # Create a subfolder per tile, so files from different ZIPs do not overwrite each other
    tile_dir = year_dir / zip_path.stem
    tile_dir.mkdir(exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tile_dir)
        print(f"Extracted: {zip_path.name} -> {tile_dir}")
    except zipfile.BadZipFile:
        print(f"Bad ZIP file: {zip_path.name}")

print("Done.")
