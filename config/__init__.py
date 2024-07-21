import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "./config.json")

# Load configuration
with open(CONFIG_PATH, "r") as config_file:
    config: dict = json.load(config_file)

LOG_PATH = config["log_path"]
DATASET_PATH = config["dataset_path"]
RANDOM_SEED = config["random_seed"]
FIGURE_PATH = config["figure_path"]
