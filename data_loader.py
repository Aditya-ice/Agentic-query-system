import pandas as pd
import json
import os
from dotenv import load_dotenv

# --- Configuration ---

# Define the paths to your data files.
# For this example, we assume they are in a 'data' subdirectory.
DATA_DIR = "datasets"
TABLE_VALUES_PATH = os.path.join(DATA_DIR, "table_values.csv")
ENCODER_SCHEMA_PATH = os.path.join(DATA_DIR, "encoder_schema.json")
DECODER_SCHEMA_PATH = os.path.join(DATA_DIR, "decoder_schema.json")
ENCODER_PARAMS_PATH = os.path.join(DATA_DIR, "encoder_parameters.json")
DECODER_PARAMS_PATH = os.path.join(DATA_DIR, "decoder_parameters.json")


# --- 1. Environment Setup: API Key ---

# Load environment variables from a .env file
load_dotenv()

# It's good practice to load the API key here, even if we don't use it in this script.
# The agent in Part 2 will need it.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in .env file.")
else:
    print("Successfully loaded GEMINI_API_KEY.")


# --- 2. Data Loading & Pre-processing ---

def load_data():
    """
    Loads all the necessary data from CSV and JSON files into memory.
    Returns a dictionary containing all the loaded data components.
    """
    print("\nLoading data...")
    try:
        # Load the main table of camera feeds into a pandas DataFrame
        camera_feeds_df = pd.read_csv(TABLE_VALUES_PATH)
        print(
            f"Successfully loaded {len(camera_feeds_df)} camera feeds from {TABLE_VALUES_PATH}")

        # Helper function to load JSON files
        def load_json(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)

        # Load the supplementary JSON data
        encoder_schema = load_json(ENCODER_SCHEMA_PATH)
        decoder_schema = load_json(DECODER_SCHEMA_PATH)
        encoder_parameters = load_json(ENCODER_PARAMS_PATH)
        decoder_parameters = load_json(DECODER_PARAMS_PATH)
        print("Successfully loaded all schema and parameter files.")

        return {
            "camera_feeds": camera_feeds_df,
            "encoder_schema": encoder_schema,
            "decoder_schema": decoder_schema,
            "encoder_parameters": encoder_parameters,
            "decoder_parameters": decoder_parameters,
        }

    except FileNotFoundError as e:
        print(
            f"Error: {e}. Make sure your data files are in a 'data' directory.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- 3. Semantic-to-Technical Mapping ---


# This dictionary translates natural language concepts into technical, queryable terms.
# This will be injected into the agent's prompt in Phase 2.
SEMANTIC_MAPPING = {
    "concepts": [
        {
            "name": "best clarity",
            "description": "A combination of high resolution and high bitrate.",
            "implementation": "Filter for resolution of '1080p' or '4K' and a bitrate greater than 5000 kbps."
        },
        {
            "name": "high efficiency",
            "description": "Refers to cameras using a modern codec like H.265, which provides good quality at lower bitrates.",
            "implementation": "Filter for codec equal to 'H.265'."
        },
        {
            "name": "region",
            "description": "The geographical area where the camera is located.",
            "implementation": "Maps directly to the 'location' column in the camera feeds table."
        }
    ]
}

print("\nDefined semantic mapping dictionary.")

# --- Main execution block ---

if __name__ == "__main__":
    # This block will run when you execute the script directly (e.g., `python data_loader.py`)
    # It demonstrates that the data loading and setup are working correctly.

    all_data = load_data()

    if all_data:
        print("\n--- Data Loading Successful! ---")
        print("\nFirst 5 rows of Camera Feeds DataFrame:")
        print(all_data["camera_feeds"].head())

        print("\nEncoder Schema Keys:")
        print(list(all_data["encoder_schema"].keys()))

        print("\nSemantic Mapping for 'best clarity':")
        print(SEMANTIC_MAPPING['concepts'][0])
        print("\n---------------------------------")
        print(
            "\nSetup for Phase 1 is complete. You are ready to build the agent in Phase 2.")
