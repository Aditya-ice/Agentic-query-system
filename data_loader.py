import pandas as pd
import json
import os
from dotenv import load_dotenv

# --- Configuration ---
# All file paths have been updated to match your specific dataset files.
DATA_DIR = "datasets"
TABLE_VALUES_PATH = os.path.join(
    DATA_DIR, "Table_feeds_v2.csv")  # CORRECTED FILENAME
TABLE_DEFS_PATH = os.path.join(
    DATA_DIR, "Table_defs_v2.csv")   # CORRECTED FILENAME
ENCODER_SCHEMA_PATH = os.path.join(DATA_DIR, "encoder_schema.json")
DECODER_SCHEMA_PATH = os.path.join(DATA_DIR, "decoder_schema.json")
ENCODER_PARAMS_PATH = os.path.join(
    DATA_DIR, "encoder_params.json")  # CORRECTED FILENAME
DECODER_PARAMS_PATH = os.path.join(
    DATA_DIR, "decoder_params.json")  # CORRECTED FILENAME

# --- Environment Setup ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in .env file. The agent will not work without it.")

# --- Data Transformation Functions ---


def create_resolution_string(row):
    """Maps width and height to a common resolution string like '1080p'."""
    if row['res_w'] == 3840 and row['res_h'] == 2160:
        return '4K'
    if row['res_w'] == 2560 and row['res_h'] == 1440:
        return '1440p'
    if row['res_w'] == 1920 and row['res_h'] == 1080:
        return '1080p'
    if row['res_w'] == 1280 and row['res_h'] == 720:
        return '720p'
    if row['res_w'] == 640 and row['res_h'] == 480:
        return '480p'
    return 'unknown'


def estimate_bitrate(row):
    """
    Estimates a proxy bitrate in kbps since it's missing from the data.
    This is a simplified heuristic: (width * height * framerate * motion_factor) / compression_ratio.
    """
    estimated_bitrate = (row['res_w'] * row['res_h']
                         * row['frrate'] * 0.07) / 1024
    return int(estimated_bitrate)

# --- Main Data Loading Function ---


def load_data():
    """
    Loads and transforms all data from source files to be ready for the agent.
    This includes renaming columns, creating new features, and loading definitions.
    """
    print("\nLoading data...")
    try:
        camera_feeds_df = pd.read_csv(TABLE_VALUES_PATH)

        camera_feeds_df.columns = [col.lower()
                                   for col in camera_feeds_df.columns]
        camera_feeds_df = camera_feeds_df.rename(
            columns={'theater': 'location'})
        camera_feeds_df['resolution'] = camera_feeds_df.apply(
            create_resolution_string, axis=1)
        camera_feeds_df['bitrate'] = camera_feeds_df.apply(
            estimate_bitrate, axis=1)

        table_definitions_df = pd.read_csv(TABLE_DEFS_PATH).set_index('header')

        def load_json(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)

        encoder_schema = load_json(ENCODER_SCHEMA_PATH)
        decoder_schema = load_json(DECODER_SCHEMA_PATH)
        encoder_parameters = load_json(ENCODER_PARAMS_PATH)
        decoder_parameters = load_json(DECODER_PARAMS_PATH)

        print("Successfully loaded and processed all data files.")

        return {
            "camera_feeds": camera_feeds_df,
            "table_definitions": table_definitions_df,
            "encoder_schema": encoder_schema,
            "decoder_schema": decoder_schema,
            "encoder_parameters": encoder_parameters,
            "decoder_parameters": decoder_parameters,
        }

    except FileNotFoundError as e:
        print(
            f"FATAL ERROR: {e}. Make sure your data files exist and the paths are correct in data_loader.py.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return None


# --- Semantic-to-Technical Mapping ---
SEMANTIC_MAPPING = {
    "concepts": [
        {
            "name": "best clarity",
            "description": "A combination of high resolution (1080p or better) and high estimated bitrate (above 4000 kbps).",
            "implementation": "Use the find_cameras tool. Filter for min_resolution of '1080p' and a min_bitrate greater than 4000."
        },
        {
            "name": "high efficiency",
            "description": "Refers to cameras using a modern, efficient codec like H265 or AV1.",
            "implementation": "Use the find_cameras tool. Filter for codec equal to 'H265' or 'AV1'."
        },
        {
            "name": "region or theater",
            "description": "The geographical area where the camera is located.",
            "implementation": "This maps to the 'location' parameter in the find_cameras tool (e.g., 'PAC', 'EUR')."
        }
    ]
}
