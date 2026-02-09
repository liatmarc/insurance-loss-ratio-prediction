"""
Data Download Helper for Insurance Loss Ratio Prediction Project

This script helps you download real insurance datasets from various sources.
Run this before starting the Jupyter notebook.
"""

import os
import urllib.request
import pandas as pd

def create_data_directory():
    """Create data directory if it doesn't exist"""
    if not os.path.exists('data'):
        os.makedirs('data')
        print("✓ Created 'data' directory")
    else:
        print("✓ 'data' directory already exists")

def download_uci_auto_insurance():
    """Download UCI Automobile dataset"""
    print("\n" + "="*60)
    print("Downloading UCI Automobile Insurance Dataset...")
    print("="*60)
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
    
    column_names = [
        'symboling', 'normalized_losses', 'make', 'fuel_type', 'aspiration',
        'num_doors', 'body_style', 'drive_wheels', 'engine_location',
        'wheel_base', 'length', 'width', 'height', 'curb_weight',
        'engine_type', 'num_cylinders', 'engine_size', 'fuel_system',
        'bore', 'stroke', 'compression_ratio', 'horsepower', 'peak_rpm',
        'city_mpg', 'highway_mpg', 'price'
    ]
    
    try:
        df = pd.read_csv(url, names=column_names, na_values='?')
        output_path = 'data/uci_automobile.csv'
        df.to_csv(output_path, index=False)
        print(f"✓ Downloaded {len(df)} records")
        print(f"✓ Saved to: {output_path}")
        print(f"  Features: {len(df.columns)}")
        print(f"  Target: 'normalized_losses' (similar to loss ratio)")
        return True
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        return False

def download_kaggle_instructions():
    """Print instructions for Kaggle datasets"""
    print("\n" + "="*60)
    print("Kaggle Insurance Claims Dataset")
    print("="*60)
    print("\nThis dataset requires a Kaggle account.")
    print("\nSteps to download:")
    print("1. Go to: https://www.kaggle.com/datasets/litvinenko630/insurance-claims")
    print("2. Click 'Download' button")
    print("3. Extract and save as: data/insurance_claims.csv")
    print("\nDataset info:")
    print("  - 1,000 insurance policies")
    print("  - 40 features (driver, vehicle, policy, claims)")
    print("  - Perfect for loss ratio prediction")

def setup_notebook_instructions():
    """Print instructions for modifying the notebook"""
    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)
    print("\n1. If you downloaded UCI Automobile data:")
    print("   - Open insurance_loss_ratio_prediction.ipynb")
    print("   - Find 'Step 1: Setup and Data Loading'")
    print("   - Uncomment and use:")
    print("     df = pd.read_csv('data/uci_automobile.csv')")
    print("\n2. If you downloaded Kaggle data:")
    print("   - Place CSV in data/ folder")
    print("   - Modify notebook to load it")
    print("\n3. Run all cells in order")
    print("\n4. Model training takes ~5-10 minutes")

def main():
    """Main function"""
    print("="*60)
    print("INSURANCE DATASET DOWNLOADER")
    print("="*60)
    
    # Create directory
    create_data_directory()
    
    # Download available datasets
    print("\nAttempting automatic downloads...")
    
    # UCI dataset
    uci_success = download_uci_auto_insurance()
    
    # Kaggle requires manual download
    download_kaggle_instructions()
    
    # Next steps
    setup_notebook_instructions()
    
    print("\n" + "="*60)
    if uci_success:
        print("✓ Setup complete! You can now run the notebook.")
    else:
        print("⚠ Some downloads failed. Check instructions above.")
    print("="*60)

if __name__ == "__main__":
    main()
