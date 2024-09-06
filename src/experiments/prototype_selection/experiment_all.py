import time
import numpy as np
from src.utils.result import log_result
from src.utils.data_preprocessing import load_data
from src.utils.evaluation_metrics import compare_prototype_selection
from src.utils.excel import save_to_excel
from src.utils.result import DatasetResult, AlgorithmResult

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

k = 1
n_folds = 10

datasets = [
    "appendicitis",
    "bupa",
    # "circles_0.05_150",
    "ecoli",
    "glass",
    "haberman",
    "heart",
    "ionosphere",
    "iris",
    "liver",
    # "moons_0.15_150",
    "movement_libras",
    "promoters",
    "sonar",
    "wine",
    "zoo",
]

algorithms = {
    "MPS": {"algorithm": MPS().fit_transform},
    "CNN": {"algorithm": CNN().fit_transform},
    "DROP3": {"algorithm": DROP3().fit_transform},
    "HMNEI": {"algorithm": HMNEI().fit_transform},
    "ICF": {"algorithm": ICF().fit_transform},
    "LDIS": {"algorithm": LDIS().fit_transform},
    "LSBo": {"algorithm": LSBo().fit_transform},
    "LSSm": {"algorithm": LSSm().fit_transform},
    "NNGIR": {"algorithm": NNGIR().fit_transform},
    "RIS1": {"algorithm": RIS("RIS1").fit_transform},
    "RIS2": {"algorithm": RIS("RIS2").fit_transform},
    "RIS3": {"algorithm": RIS("RIS3").fit_transform},
}

tmp_results: list[DatasetResult] = []
excel_content = {}
folder = f"prototype_selection/{time.strftime("%Y-%m-%d %H-%M-%S")}_{n_folds}-fold_{len(datasets)}-DS"+ "/"

for dataset_name in tqdm.tqdm(datasets, desc="Dataset progress", leave=False):
    bt_ds = dataset_name.replace("_", " ").capitalize()
    file_name = folder + f"prototype selection {n_folds}-fold {bt_ds}"
    X, y = load_data(dataset_name)
    tmp_result = compare_prototype_selection(X, y, algorithms, 1, n_folds)
    tmp_result.dataset = dataset_name
    print(f"Dataset: {dataset_name}")
    log_result(tmp_result, file_name)
    print("\n")

    tmp_results.append(tmp_result)
    tmp_ecxel_content = tmp_result.ecxel_content()
    save_to_excel({bt_ds: tmp_ecxel_content}, file_name)
    excel_content[bt_ds] = tmp_ecxel_content

final_res = DatasetResult("Final results", n_folds, k)

for algorithm_name in tmp_results[0].results.keys():
    final_res.add_result(AlgorithmResult(algorithm_name))
    for dataset_res in tmp_results:
        final_res.results[algorithm_name].add_result(
            dataset_res.results[algorithm_name].result
        )


excel_content["Final results"] = final_res.ecxel_content()

print("Final results:")
file_name= folder + f"prototype selection {n_folds}-fold {len(datasets)}-DS"
log_result(final_res, file_name)
save_to_excel(
    excel_content,
    file_name
)
