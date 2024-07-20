import random
from src.utils.evaluation_metrics import compare_prototype_selection
from src.algorithms.drop3 import DROP3
from src.algorithms.notWork_drlsh import DRLSH
from src.utils.result import log_result
from sklearn.datasets import load_digits

# set random seed to 42
random.seed(42)

# Log the results
log_path = "results/logs/experiment_drlsh.log"

# Load diload_digits dataset
data = load_digits()
X, y = data.data, data.target

# Scale the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Define the algorithms
algorithms = {
    "DROP3": {"algorithm": DROP3().fit_transform},
    # "RIS1": {"algorithm": RIS("RIS1", 0.1).fit_transform},
    # "RIS2": {"algorithm": RIS("RIS2", 0.1).fit_transform},
    # "RIS3": {"algorithm": RIS("RIS3", 0.1).fit_transform},
    "DRLSH": {"algorithm": DRLSH().fit_transform},
}

result = compare_prototype_selection(X, y, algorithms, 3, 10)

log_result(result, log_path, "digits")
