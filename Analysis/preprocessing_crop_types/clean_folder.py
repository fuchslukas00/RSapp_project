from pathlib import Path
import shutil

# Root folder from previous extraction
src_root = Path(r"C:\Users\lukas\Documents\Studium\Remote_Sensing_Products\project\crop_yields\data\crop_types_D\crop_types\Results\extracted_by_year")

# New folder for only TIF files
out_root = src_root.parent / "tifs_by_year"
out_root.mkdir(exist_ok=True)

for year_dir in src_root.iterdir():
    if not year_dir.is_dir():
        continue

    year = year_dir.name
    out_year_dir = out_root / year
    out_year_dir.mkdir(exist_ok=True)

    # find all tif files recursively
    tif_files = list(year_dir.rglob("*.tif"))

    print(f"{year}: found {len(tif_files)} tif files")

    for tif in tif_files:
        destination = out_year_dir / tif.name
        
        # avoid overwriting
        if destination.exists():
            destination = out_year_dir / f"{tif.stem}_{tif.parent.name}.tif"
        
        shutil.copy2(tif, destination)

print("Done collecting tif files.")
