from src.utils.result import log_result
from src.utils.data_preprocessing import load_data
from src.utils.evaluation_metrics import compare_prototype_selection

from src.algorithms.cnn import CNN
from src.algorithms.drop3 import DROP3
from src.algorithms.icf import ICF
from src.algorithms.ldis import LDIS
from src.algorithms.lsbo import LSBo
from src.algorithms.lssm import LSSm
from src.algorithms.ris import RIS
from src.algorithms.hmnei import HMNEI
from src.algorithms.nngir import NNGIR

DATASET_NAME = "wine"

# Get file name
FILE_NAME = __file__.split("/")[-1].split(".")[0]

# Load dataset
X, y = load_data(DATASET_NAME)

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
    "NNGIR": {"algorithm": NNGIR().fit_transform},
}

result = compare_prototype_selection(X, y, algorithms, 1, 10)

log_result(result, FILE_NAME, DATASET_NAME)
