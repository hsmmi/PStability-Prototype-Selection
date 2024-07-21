from src.utils.result import log_result
from src.utils.data_preprocessing import load_data
from src.utils.evaluation_metrics import compare_prototype_selection

from src.algorithms.drop3 import DROP3
from src.algorithms.cnnir import CNNIR


DATASET_NAME = "wine"

# Get file name
FILE_NAME = __file__.split("/")[-1].split(".")[0]

# Load dataset
X, y = load_data(DATASET_NAME)

# Define the algorithms
algorithms = {
    "DROP3": {"algorithm": DROP3().fit_transform},
    "CNNIR": {"algorithm": CNNIR().fit_transform},
}

result = compare_prototype_selection(X, y, algorithms, 3, 10)

log_result(result, FILE_NAME, DATASET_NAME)
