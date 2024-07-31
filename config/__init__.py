# Import the rest of the configuration after logging setup
from config.config_loader import loaded_config

# from config.log import setup_logging

# Now you can use loaded_config and LOGGING_CONFIG as needed
LOG_PATH = loaded_config["log_path"]
DATASET_PATH = loaded_config["dataset_path"]
RANDOM_SEED = loaded_config["random_seed"]
FIGURE_PATH = loaded_config["figure_path"]
LOGGING_CONFIG = loaded_config["logging_config"]

# Set up logging
# setup_logging()
