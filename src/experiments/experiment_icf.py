import random
from src.utils.evaluation_metrics import compare_prototype_selection
from src.algorithms.drop3 import DROP3
from src.algorithms.icf import ICF
from sklearn.datasets import load_digits
from src.utils.result import log_result

# set random seed to 42
random.seed(42)

# Log the results
log_path = "results/logs/experiment_icf.log"

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
    # "RIS1": {"algorithm": RIS("RIS1").fit_transform},
    "ICF": {"algorithm": ICF().fit_transform},
    "ICF2": {"algorithm": ICF2().select_data},
}

result = compare_prototype_selection(X, y, algorithms, 3, 10)

log_result(result, log_path, "digits")
