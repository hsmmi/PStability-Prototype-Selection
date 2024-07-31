from src.utils.result import log_result
from src.utils.data_preprocessing import load_data
from src.utils.evaluation_metrics import compare_prototype_selection

from src.algorithms.prototype_selection.drop3 import DROP3
from src.algorithms.prototype_selection.nngir import NNGIR
from src.algorithms.prototype_selection.hmnei import HMNEI

DATASET_NAME = "banana"

# Get file name
from src.utils.path import ProjectPath

FILE_NAME = ProjectPath(__file__).get_safe_filename()

# Load dataset
X, y = load_data(DATASET_NAME)

# Define the algorithms
algorithms = {
    # "DROP3": {"algorithm": DROP3().fit_transform},
    "NNGIR": {"algorithm": NNGIR().fit_transform},
    # "HMNEI": {"algorithm": HMNEI().fit_transform},
}

result = compare_prototype_selection(X, y, algorithms, 1, 10)

log_result(result, FILE_NAME, DATASET_NAME)
