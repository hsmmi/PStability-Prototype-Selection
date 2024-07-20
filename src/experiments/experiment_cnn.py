import random

from sklearn.datasets import load_wine
from src.algorithms.cnn import CNN
from src.algorithms.drop3 import DROP3
from src.utils.evaluation_metrics import compare_prototype_selection
from src.utils.result import log_result

# set random seed to 42
random.seed(42)

# Log the results
log_path = "results/logs/experiment_cnn.log"

# Load diload_wine dataset
data = load_wine()
X, y = data.data, data.target

# Scale the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

algorithms = {
    "DROP3": {"algorithm": DROP3().fit_transform},
    "CNN": {"algorithm": CNN().fit_transform},
}

result = compare_prototype_selection(X, y, algorithms, 3, 10)

log_result(result, log_path, "wine")
