import os
import requests
import sys
import zipfile
import io

# --- Configuration ---
# Directory to save the downloaded data files
DATA_DIR = "data"

# List of datasets to download
# Each item is a dictionary with 'filename', 'url', and optionally 'extract_file'
DATASETS_TO_DOWNLOAD = [
    {
        "filename": "Concrete_Data.xls",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    },
    {
        "filename": "communities.data",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
    },
    {
        "filename": "communities.names", # Description file, good to have
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names"
    },
    {
        "filename": "Bike-Sharing-Dataset.zip", # This is the ZIP file
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
        "extract_file": "hour.csv" # We want to extract this file from the zip
    },
    {
        "filename": "parkinsons_updrs.data",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"
    }
]

# --- Helper Function ---
def download_file(url, target_path):
    """
    Downloads a file from a given URL to a target path.
    """
    try:
        response = requests.get(url, stream=True, timeout=30) # Added timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        print(f"Successfully downloaded {os.path.basename(target_path)}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        # Clean up partially downloaded file if it exists
        if os.path.exists(target_path):
            os.remove(target_path)
        return False
    except IOError as e:
        print(f"Error writing file {target_path}: {e}", file=sys.stderr)
        return False

def extract_zip_member(zip_content_bytes, member_name, target_dir):
    """
    Extracts a specific member from a ZIP file (provided as bytes) to a target directory.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_content_bytes)) as zf:
            # Check if member exists
            if member_name not in zf.namelist():
                print(f"Error: {member_name} not found in the zip file.", file=sys.stderr)
                return False
            zf.extract(member_name, target_dir)
            print(f"Successfully extracted {member_name} to {target_dir}")
            return True
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid ZIP file or is corrupted.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error extracting {member_name} from zip: {e}", file=sys.stderr)
        return False

# --- Main Download Logic ---
def main():
    """
    Main function to create data directory and download datasets.
    """
    # Create the data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR)
            print(f"Created directory: {DATA_DIR}")
        except OSError as e:
            print(f"Error creating directory {DATA_DIR}: {e}", file=sys.stderr)
            return # Exit if directory creation fails

    # Download each dataset
    all_successful = True
    for dataset_info in DATASETS_TO_DOWNLOAD:
        filename = dataset_info["filename"]
        url = dataset_info["url"]
        file_path = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(file_path) or (dataset_info.get("extract_file") and not os.path.exists(os.path.join(DATA_DIR, dataset_info["extract_file"]))):
            print(f"Downloading {filename} from {url}...")
            success = download_file(url, file_path)
            if success:
                # If an extract_file is specified, attempt to extract it
                extract_target_name = dataset_info.get("extract_file")
                if extract_target_name:
                    extracted_file_path = os.path.join(DATA_DIR, extract_target_name)
                    if not os.path.exists(extracted_file_path):
                        print(f"Attempting to extract {extract_target_name} from {filename}...")
                        # Read the downloaded zip file bytes for extraction
                        with open(file_path, 'rb') as f_zip:
                            zip_bytes = f_zip.read()
                        if not extract_zip_member(zip_bytes, extract_target_name, DATA_DIR):
                            all_successful = False
                            print(f"Could not extract {extract_target_name}. You may need to extract it manually into the '{DATA_DIR}' directory.", file=sys.stderr)
                        # Optionally, remove the zip file after successful extraction if desired
                        # os.remove(file_path)
                        # print(f"Removed {filename} after extraction.")
                    else:
                        print(f"{extract_target_name} already exists in {DATA_DIR}. Skipping extraction.")
            else:
                all_successful = False
        else:
            print(f"{filename} already exists in {DATA_DIR}. Skipping download.")
            
    if all_successful:
        print("\nAll required data files are checked/downloaded.")
    else:
        print("\nSome files could not be downloaded. Please check the error messages above.", file=sys.stderr)

if __name__ == "__main__":
    main() 