from src.utils.visualization import plot_algorithm_results
from src.utils.data_preprocessing import load_data

from src.algorithms.cnn import CNN
from src.algorithms.drop3 import DROP3
from src.algorithms.icf import ICF
from src.algorithms.ldis import LDIS
from src.algorithms.lsbo import LSBo
from src.algorithms.lssm import LSSm
from src.algorithms.ris import RIS
from src.algorithms.hmnei import HMNEI
from src.algorithms.notWork_nngir import NNGIR

import tqdm

datasets = ["circles_0.05", "moons_0.15", "banana"]

# Define the algorithms
algorithms = {
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

# progress bar
for dataset_name in tqdm.tqdm(datasets, desc="Dataset progress", leave=False):
    for algorithm_name, algorithm in tqdm.tqdm(
        algorithms.items(), desc="Algorithm progress", leave=False
    ):
        X, y = load_data(dataset_name)
        X_, y_ = algorithm["algorithm"](X, y)
        plot_algorithm_results(
            X=X,
            y=y,
            X_=X_,
            y_=y_,
            title=f"{dataset_name}_{algorithm_name}",
            save_plot=True,
            show_plot=False,
        )
