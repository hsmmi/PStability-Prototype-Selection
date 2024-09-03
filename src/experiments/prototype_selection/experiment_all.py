import time
import numpy as np
from src.utils.result import log_result
from src.utils.data_preprocessing import load_data
from src.utils.evaluation_metrics import compare_prototype_selection
from src.utils.excel import save_to_excel

from src.algorithms.prototype_selection.cnn import CNN
from src.algorithms.prototype_selection.drop3 import DROP3
from src.algorithms.prototype_selection.icf import ICF
from src.algorithms.prototype_selection.ldis import LDIS
from src.algorithms.prototype_selection.lsbo import LSBo
from src.algorithms.prototype_selection.lssm import LSSm
from src.algorithms.prototype_selection.ris import RIS
from src.algorithms.prototype_selection.hmnei import HMNEI
from src.algorithms.prototype_selection.nngir import NNGIR
from src.algorithms.stability.my_prototype_selection import PrototypeSelection as MPS

import tqdm

# Get file name
from src.utils.path import ProjectPath

FILE_NAME = ProjectPath(__file__).get_safe_filename()

n_folds = 10

datasets = [
    "ecoli",
    "ionosphere",
    "heart",
    "sonar",
    "haberman",
    "liver",
    "iris",
    "wine",
    "moons_0.15_150",
    "circles_0.05_150",
    "zoo",
    "glass",
    "promoters",
]
# datasets = ["iris"]

algorithms = {
    "MPS": {"algorithm": MPS().fit_transform},
    "CNN": {"algorithm": CNN().fit_transform},
    "DROP3": {"algorithm": DROP3().fit_transform},
    "ICF": {"algorithm": ICF().fit_transform},
    "LDIS": {"algorithm": LDIS().fit_transform},
    "LSBo": {"algorithm": LSBo().fit_transform},
    "LSSm": {"algorithm": LSSm().fit_transform},
    "RIS1": {"algorithm": RIS("RIS1").fit_transform},
    "RIS2": {"algorithm": RIS("RIS2").fit_transform},
    "RIS3": {"algorithm": RIS("RIS3").fit_transform},
    "HMNEI": {"algorithm": HMNEI().fit_transform},
    "NNGIR": {"algorithm": NNGIR().fit_transform},
}

tmp_results = []
excel_content = {}
folder = f"prototype selection {n_folds}-fold {len(datasets)}-DS {time.strftime("%Y-%m-%d %H:%M:%S")}"+ "/"

for dataset_name in tqdm.tqdm(datasets, desc="Dataset progress", leave=False):
    X, y = load_data(dataset_name)
    tmp_result = compare_prototype_selection(X, y, algorithms, 1, n_folds)
    print(f"Dataset: {dataset_name}")
    log_result(tmp_result, FILE_NAME, dataset_name)
    print("\n")
    tmp_results.append(tmp_result)
    tmp_content = {
        dataset_name: {
            "Algorithms": list(tmp_result.keys()),
            "Acc. Train": [f"{tmp_result[key][0]:.2%}" for key in tmp_result],
            "Acc. Test": [f"{tmp_result[key][1]:.2%}" for key in tmp_result],
            "Size": [f"{tmp_result[key][2]:.2f}" for key in tmp_result],
            "Distortion": [f"{tmp_result[key][3]:.2f}" for key in tmp_result],
            "Objective Function": [f"{tmp_result[key][4]:.2f}" for key in tmp_result],
            "Reduction": [f"{tmp_result[key][5]:.2%}" for key in tmp_result],
            "Time": [f"{tmp_result[key][6]:.3f}" for key in tmp_result],
        }
    }
    save_to_excel(tmp_content, folder + f"prototype selection {n_folds}-fold {dataset_name}")
    excel_content[dataset_name] = tmp_content[dataset_name]

result = {}
for algorithm_name, _ in tmp_results[0].items():
    result[algorithm_name] = []
    for tmp_result in tmp_results:
        result[algorithm_name].append(tmp_result[algorithm_name])

    result[algorithm_name] = np.array(result[algorithm_name]).mean(axis=0)

excel_content["Final results"] = {
    "Algorithms": list(result.keys()),
    "Acc. Train": [f"{result[key][0]:.2%}" for key in result],
    "Acc. Test": [f"{result[key][1]:.2%}" for key in result],
    "Size": [f"{result[key][2]:.2f}" for key in result],
    "Distortion": [f"{result[key][3]:.2f}" for key in result],
    "Objective Function": [f"{result[key][4]:.2f}" for key in result],
    "Reduction": [f"{result[key][5]:.2%}" for key in result],
    "Time": [f"{result[key][6]:.3f}" for key in result],
}

print("Final results:")
log_result(result, FILE_NAME, datasets)
save_to_excel(
    excel_content,
    folder
    + f"prototype selection {n_folds}-fold {len(datasets)}-DS",
)
