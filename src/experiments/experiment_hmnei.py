import random
from src.utils.evaluation_metrics import compare_prototype_selection
from src.algorithms.drop3 import DROP3
from src.algorithms.hmnei import HMNEI
from sklearn.datasets import load_digits
from src.utils.result import log_result

# set random seed to 42
random.seed(42)

# Log the results
log_path = "results/logs/experiment_hmnei.log"

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
    "HMNEI": {"algorithm": HMNEI().fit_transform},
}

result = compare_prototype_selection(X, y, algorithms, 3, 10)

log_result(result, log_path, "digits")
