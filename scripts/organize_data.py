import os
import shutil
from pathlib import Path
from collections import defaultdict

# ------------------------------------------------------------------------------------------------
# rclone script
# rclone copy gdrive:"MQP Data" ./raw_data_new_ --drive-shared-with-me --include "*20x*100*" --include "*60x*160*" --progress
# ------------------------------------------------------------------------------------------------

def reorganize_raw_data_paired(source_dir, output_dir, dry_run=True):
    """
    Reorganize raw data into biofilm/ and release/ folders with only paired images.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    # Scan files to catalog all images
    image_registry = defaultdict(dict)
    print("=== Scanning files ===\n")


    for date_folder in sorted(source_path.iterdir()):
        if not date_folder.is_dir():
            continue

        date = date_folder.name  # e.g., "06-08-2025"

        for treatment_mag_folder in sorted(date_folder.iterdir()):
            if not treatment_mag_folder.is_dir():
                continue

            folder_name = treatment_mag_folder.name  # e.g., "DNaseI_20x"

            # Extract treatment and magnification
            if "_20x" in folder_name:
                treatment = folder_name.replace("_20x", "")
                image_type = "biofilm"
            elif "_60x" in folder_name:

                treatment = folder_name.replace("_60x", "")
                image_type = "release"
            else:
                print(f"Warning: Unknown magnification in {folder_name}")
                continue

            for xy_folder in sorted(treatment_mag_folder.iterdir()):
                if not xy_folder.is_dir():
                    continue

                xy_name = xy_folder.name  # e.g., "XY01"

                # Extract XY number
                if xy_name.startswith("XY"):

                    xy_num = int(xy_name[2:])  # Convert "XY01" -> 1
                else:
                    print(f"Warning: Unknown XY folder format {xy_name}")
                    continue

                # Process each .tif file in the XY folder
                for tif_file in xy_folder.glob("*.tif"):
                    key = (treatment, date, xy_num)
                    image_registry[key][image_type] = tif_file

    # Identify complete pairs
    complete_pairs = []
    incomplete_images = []

    for key, images in sorted(image_registry.items()):

        treatment, date, xy_num = key
        if "biofilm" in images and "release" in images:
            complete_pairs.append((key, images))
        else:
            missing = "release" if "biofilm" in images else "biofilm"
            has = "biofilm" if "biofilm" in images else "release"
            incomplete_images.append((key, has, missing))

    print(f"Found {len(complete_pairs)} complete pairs")
    print(f"Found {len(incomplete_images)} incomplete images (will be skipped)\n")

    if incomplete_images:
        print("=== Incomplete images (skipped) ===")
        for key, has, missing in incomplete_images:
            treatment, date, xy_num = key
            print(f"  {treatment}_{date}_{xy_num}: has {has}, missing {missing}")
        print()

    # Copy complete pairs
    if not dry_run:
        (output_path / "biofilm").mkdir(parents=True, exist_ok=True)

        (output_path / "release").mkdir(parents=True, exist_ok=True)

    print("=== Complete pairs to process ===\n")

    for key, images in complete_pairs:
        treatment, date, xy_num = key

        # Process biofilm image
        biofilm_file = images["biofilm"]
        biofilm_new_name = f"biofilm_{treatment}_{date}_{xy_num}.tif"
        biofilm_new_path = output_path / "biofilm" / biofilm_new_name

        # Process release image
        release_file = images["release"]

        release_new_name = f"release_{treatment}_{date}_{xy_num}.tif"
        release_new_path = output_path / "release" / release_new_name

        if dry_run:
            print(f"Pair: {treatment}_{date}_{xy_num}")
            print(
                f"  Biofilm: {biofilm_file.relative_to(source_path)} -> biofilm/{biofilm_new_name}"
            )
            print(
                f"  Release: {release_file.relative_to(source_path)} -> release/{release_new_name}"
            )
            print()
        else:
            shutil.copy2(biofilm_file, biofilm_new_path)
            shutil.copy2(release_file, release_new_path)
            print(f"Copied pair: {treatment}_{date}_{xy_num}")

    print(f"\n{'DRY RUN ' if dry_run else ''}Summary:")
    print(f"Complete pairs processed: {len(complete_pairs)}")
    print(f"Images skipped (no pair): {len(incomplete_images)}")

    if dry_run:
        print("\nThis was a DRY RUN. No files were copied.")
        print("Set dry_run=False to actually copy the files.")
    else:
        print(f"\nFiles organized in:")
        print(f"  {output_path / 'biofilm'}")
        print(f"  {output_path / 'release'}")
        print("\nFiles are sorted identically in both folders for easy pairing.")


if __name__ == "__main__":
    # Configure paths
    source_directory = "data/raw"
    output_directory = "data/processed"

    # First, run with dry_run=True to see what will happen
    print("=== DRY RUN ===\n")
    reorganize_raw_data_paired(source_directory, output_directory, dry_run=True)

    # Uncomment the following lines to actually perform the reorganization
    # print("\n\n=== ACTUAL RUN ===\n")
    # reorganize_raw_data_paired(source_directory, output_directory, dry_run=False)
