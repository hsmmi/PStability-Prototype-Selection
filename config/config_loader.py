"""
This module loads configuration from a JSON file and provides a global variable
"""

import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


def load_config(config_path: str) -> dict:
    """Load configuration from a JSON file."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as config_file:
        try:
            loaded_config = json.load(config_file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON in configuration file: {e}")

    required_keys = [
        "log_path",
        "dataset_path",
        "random_seed",
        "figure_path",
        "logging_config",
    ]
    for key in required_keys:
        if key not in loaded_config:
            raise KeyError(f"Missing required configuration key: {key}")

    return loaded_config


# Load configuration
try:
    loaded_config = load_config(CONFIG_PATH)
except (FileNotFoundError, ValueError, KeyError) as e:
    print(f"Configuration error: {e}")
    raise  # Re-raise the exception to halt the application
