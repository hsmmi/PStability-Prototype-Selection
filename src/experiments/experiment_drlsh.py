from src.utils.data_preprocessing import load_data
from src.utils.evaluation_metrics import compare_prototype_selection
from src.algorithms.drop3 import DROP3
from src.algorithms.drlsh import DRLSH
from src.utils.result import log_result

DATASET_NAME = "moons_0.15"

# Get file name
FILE_NAME = __file__.split("/")[-1].split(".")[0]

# Load dataset
X, y = load_data(DATASET_NAME)

# Define the algorithms
algorithms = {
    "DRLSH": {"algorithm": DRLSH(L=30, M=10).fit_transform},
    "DROP3": {"algorithm": DROP3().fit_transform},
    # "RIS1": {"algorithm": RIS("RIS1", 0.1).fit_transform},
    # "RIS2": {"algorithm": RIS("RIS2", 0.1).fit_transform},
    # "RIS3": {"algorithm": RIS("RIS3", 0.1).fit_transform},
}

result = compare_prototype_selection(X, y, algorithms, 1, 10)

log_result(result, FILE_NAME, DATASET_NAME)
