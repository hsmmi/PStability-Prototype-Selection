from src.utils.visualization import plot_algorithm_results
from src.utils.data_preprocessing import load_data

from src.algorithms.stability.my_prototype_selection import PrototypeSelection as MPS
from src.algorithms.prototype_selection.enn import ENN
from src.algorithms.prototype_selection.cnn import CNN
from src.algorithms.prototype_selection.drlsh import DRLSH
from src.algorithms.prototype_selection.drop3 import DROP3
from src.algorithms.prototype_selection.icf import ICF
from src.algorithms.prototype_selection.ldis import LDIS
from src.algorithms.prototype_selection.lsbo import LSBo
from src.algorithms.prototype_selection.lssm import LSSm
from src.algorithms.prototype_selection.ris import RIS
from src.algorithms.prototype_selection.hmnei import HMNEI
from src.algorithms.prototype_selection.nngir import NNGIR

import tqdm

datasets = ["banana_undersampled"]

# Define the algorithms
algorithms = {
    "ENN": {"algorithm": ENN().fit_transform},
    "CNN": {"algorithm": CNN().fit_transform},
    "DRLSH": {"algorithm": DRLSH().fit_transform},
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
    "PSPS": {"algorithm": MPS().fit_transform},
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
            title=f"{algorithm_name}",
            save_plot=True,
            show_plot=False,
        )
