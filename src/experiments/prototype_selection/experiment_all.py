import numpy as np
from src.utils.result import log_result
from src.utils.data_preprocessing import load_data
from src.utils.evaluation_metrics import compare_prototype_selection

from src.algorithms.prototype_selection.cnn import CNN
from src.algorithms.prototype_selection.drop3 import DROP3
from src.algorithms.prototype_selection.icf import ICF
from src.algorithms.prototype_selection.ldis import LDIS
from src.algorithms.prototype_selection.lsbo import LSBo
from src.algorithms.prototype_selection.lssm import LSSm
from src.algorithms.prototype_selection.ris import RIS
from src.algorithms.prototype_selection.hmnei import HMNEI
from src.algorithms.prototype_selection.nngir import NNGIR
import tqdm

# Get file name
from src.utils.path import ProjectPath

FILE_NAME = ProjectPath(__file__).get_safe_filename()

datasets = ["iris", "wine", "moons_0.15", "circles_0.05"]

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

tmp_results = []

for dataset_name in tqdm.tqdm(datasets, desc="Dataset progress", leave=False):
    X, y = load_data(dataset_name)

    tmp_results.append(compare_prototype_selection(X, y, algorithms, 1, 10))

result = {}
for algorithm_name, _ in tmp_results[0].items():
    result[algorithm_name] = []
    for tmp_result in tmp_results:
        result[algorithm_name].append(tmp_result[algorithm_name])

    result[algorithm_name] = np.array(result[algorithm_name]).mean(axis=0)

log_result(result, FILE_NAME, datasets)
