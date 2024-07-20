import random

from sklearn.datasets import load_digits
from src.algorithms.cnn import CNN
from src.algorithms.drop3 import DROP3
from src.algorithms.icf import ICF
from src.algorithms.ldis import LDIS
from src.algorithms.lsbo import LSBo
from src.algorithms.lssm import LSSm
from src.algorithms.ris import RIS
from src.algorithms.hmnei import HMNEI
from src.utils.evaluation_metrics import compare_prototype_selection
from src.utils.result import log_result

# set random seed to 42
random.seed(42)

# Log the results
log_path = "results/logs/experiment_all.log"

# Load diload_digits dataset
data = load_digits()
X, y = data.data, data.target

# Scale the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

algorithms = {
    "CNN": {"algorithm": CNN().fit_transform},
    "DROP3": {"algorithm": DROP3().fit_transform},
    "ICF": {"algorithm": ICF().fit_transform},
    "LDIS": {"algorithm": LDIS().fit_transform},
    "LSBo": {"algorithm": LSBo().fit_transform},
    "LSSm": {"algorithm": LSSm().fit_transform},
    # "RIS1": {"algorithm": RIS("RIS1").fit_transform},
    # "RIS2": {"algorithm": RIS("RIS2").fit_transform},
    # "RIS3": {"algorithm": RIS("RIS3").fit_transform},
    "HMNEI": {"algorithm": HMNEI().fit_transform},
}

result = compare_prototype_selection(X, y, algorithms, 3, 5)

log_result(result, log_path, "digits")
