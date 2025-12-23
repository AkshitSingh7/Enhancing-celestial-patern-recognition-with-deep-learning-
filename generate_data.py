import os
from src.fetch_data import get_gaia_catalog
from src.simulate import generate_dataset_split

CATALOG_FILE = "data/gaia_dr3_raw.fits"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"

def main():
    # 1. Fetch Data
    print("Step 1: Fetching Gaia Catalog...")
    get_gaia_catalog(CATALOG_FILE, limit=5000)

    # 2. Generate Training Data
    print("\nStep 2: Generating Training Set...")
    generate_dataset_split(CATALOG_FILE, TRAIN_DIR, num_images=200)

    # 3. Generate Validation Data
    print("\nStep 3: Generating Validation Set...")
    generate_dataset_split(CATALOG_FILE, VAL_DIR, num_images=50)

    print("\nData generation complete!")

if __name__ == "__main__":
    main()
