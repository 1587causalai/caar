import os
import sys
import numpy as np
import pandas as pd
import argparse # Added argparse

# Adjust path to import from the 'real' module
# Assuming this script (validate_data_loading.py) is in src/data/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))  # This is the 'src' directory
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(SRC_DIR, '..')) # Project root directory

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT_DIR not in sys.path: # Also add project root for potential utils etc.
    sys.path.insert(0, PROJECT_ROOT_DIR)

# Now we can import from data.real
from data.real import prepare_real_data_experiment

# Dictionary mapping dataset names (that require local files) to their expected primary file
# This helps in providing more accurate messages to the user.
DATASET_EXPECTED_FILES = {
    'concrete_strength': 'Concrete_Data.xls (or .csv)',
    'communities_crime': 'communities.data',
    'bike_sharing': 'hour.csv (extracted from Bike-Sharing-Dataset.zip)',
    'parkinsons_telemonitoring': 'parkinsons_updrs.data'
    # Add other datasets here if they require specific local files not handled by sklearn fetchers
}

SKLEARN_LOADED_DATASETS = ['california', 'diabetes', 'boston_housing'] # Uses alias 'california' for california_housing

def validate_dataset_loading(dataset_name: str):
    print(f"--- Validating Data Loading for: {dataset_name} ---")
    
    data_dir_path = os.path.join(PROJECT_ROOT_DIR, 'data')
    
    print(f"\nAttempting to load '{dataset_name}' dataset...")
    
    expected_file_info = DATASET_EXPECTED_FILES.get(dataset_name)
    if expected_file_info:
        print(f"Expected data file in '{data_dir_path}': {expected_file_info}")
    elif dataset_name in SKLEARN_LOADED_DATASETS or dataset_name == 'california_housing': # california_housing might be used directly
        print(f"This dataset ('{dataset_name}') is typically loaded via scikit-learn internal functions (e.g., fetch_openml or built-in). No specific user-downloaded file in '{data_dir_path}' is strictly required by this script for it, assuming scikit-learn can access it.")
    else:
        print(f"Warning: File expectation for '{dataset_name}' not explicitly defined in this validation script. Assuming direct load or scikit-learn fetcher.")

    try:
        X_train, X_val, X_test, \
        y_train, y_val, y_test, \
        feature_names, \
        outlier_mask_train, outlier_mask_val, \
        scaler = prepare_real_data_experiment(
            dataset_name=dataset_name,
            data_dir=data_dir_path,
            outlier_ratio=0.0, 
            test_size=0.2,    
            val_size=0.1,     
            scale=True,       
            random_state=42
        )

        print("\n--- Data Loading Successful ---")

        print("\n1. Shapes of the returned arrays:")
        print(f"   X_train shape: {X_train.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   X_val shape:   {X_val.shape}")
        print(f"   y_val shape:   {y_val.shape}")
        print(f"   X_test shape:  {X_test.shape}")
        print(f"   y_test shape:  {y_test.shape}")

        print("\n2. Feature names:")
        print(f"   Feature names ({len(feature_names)}): {feature_names}")
        if X_train.shape[1] == len(feature_names):
            print(f"   Number of feature names matches number of columns in X_train ({X_train.shape[1]}). OK.")
        else:
            print(f"   WARNING: Mismatch! Feature names count: {len(feature_names)}, X_train columns: {X_train.shape[1]}.")

        print("\n3. Data types:")
        print(f"   X_train dtype: {X_train.dtype}")
        print(f"   y_train dtype: {y_train.dtype}")

        print("\n4. Checking for NaN values in training data (after processing):")
        nan_in_X_train = np.isnan(X_train).sum()
        nan_in_y_train = np.isnan(y_train).sum()
        print(f"   NaNs in X_train: {nan_in_X_train}")
        print(f"   NaNs in y_train: {nan_in_y_train}")
        if nan_in_X_train == 0 and nan_in_y_train == 0:
            print("   NaN check: OK (No NaNs found in processed training data).")
        else:
            print("   WARNING: NaNs found in processed training data!")

        print("\n5. First 3 rows of X_train (scaled, if applicable):")
        if X_train.shape[0] >= 3:
            X_train_df_sample = pd.DataFrame(X_train[:3, :], columns=feature_names)
            print(X_train_df_sample.to_string())
        elif X_train.shape[0] > 0:
            print(X_train[:X_train.shape[0], :])
        else:
            print("   X_train is empty.")

        print("\n6. First 3 values of y_train:")
        if y_train.shape[0] >= 3:
            print(y_train[:3])
        elif y_train.shape[0] > 0:
            print(y_train[:y_train.shape[0]])
        else:
            print("   y_train is empty.")
        
        if scaler:
            print("\n7. Scaler details (if scaling was applied):")
            print(f"   Scaler object: {scaler}")
            if hasattr(scaler, 'mean_') and scaler.mean_ is not None:
                 print(f"   Scaler mean for first few features: {scaler.mean_[:min(3, len(scaler.mean_))]}")
            if hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                 print(f"   Scaler scale (std) for first few features: {scaler.scale_[:min(3, len(scaler.scale_))]}")
        else:
            print("\n7. Scaler was not used (scale=False).")

        print(f"\n--- Validation for '{dataset_name}' Finished Successfully ---")

    except FileNotFoundError as e:
        print(f"\nERROR: Data file not found. {e}")
        print(f"For datasets requiring local files (e.g., {list(DATASET_EXPECTED_FILES.keys())}), please ensure you have run 'python download_data.py' from the project root directory ('{PROJECT_ROOT_DIR}').")
        print(f"The expected file for '{dataset_name}' (if applicable) was not found in '{data_dir_path}'.")
    except ValueError as e: # Catch specific errors from prepare_real_data_experiment like unsupported dataset
        print(f"\nAN ERROR OCCURRED (ValueError): {e}")
        print(f"This might be due to an unsupported dataset name or an issue within the data preparation function for '{dataset_name}'.")
    except Exception as e:
        print(f"\nAN UNEXPECTED ERROR OCCURRED: {e}")
        import traceback
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("--- End of Traceback ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate data loading and preprocessing for a specified dataset.")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Name of the dataset to validate (e.g., 'parkinsons_telemonitoring', 'california', 'communities_crime').")
    args = parser.parse_args()
    
    validate_dataset_loading(args.dataset) 