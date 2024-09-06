import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from src.algorithms.stability.my_prototype_selection import PrototypeSelection
from src.utils.excel import save_to_excel
from config.log import get_logger
from src.utils.data_preprocessing import load_data

logger = get_logger("mylogger")
logger.setLevel("WARNING")

n_folds = 5
n_neighbors = 1
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
stability_list = [1, 3, 5, 7, 9, 11]
excel_total = {}

dataset_list = [
    "circles_0.05_150",
    "moons_0.15_150",
    "iris_undersampled",
    "wine_undersampled",
    "iris_0_1",
    "iris_0_2",
    "iris_1_2",
    "iris",
    "wine",
]


def run_stability(stability: int):
    excel_content = {}

    prototype_selection = PrototypeSelection()

    for dataset in tqdm(dataset_list, desc="Datasets progress", leave=False):
        X, y = load_data(dataset)

        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        results = {"total_distortions": [], "accuracy": []}
        min_len = len(X)
        for train_index, test_index in tqdm(
            kf.split(X, y), total=n_folds, desc="K-Fold progress", leave=False
        ):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            prototype_selection.fit(X_train, y_train)

            result = prototype_selection.prototype_reduction(stability)

            removed_prototypes = result["removed_prototypes"]
            base_total_distortion = result["base_total_distortion"]
            idx_min_total_distortion = result["idx_min_total_distortion"]
            last_idx_under_base = result["last_idx_under_base"]

            accuracy = []

            remain_prototypes = list(range(len(X_train)))
            for idx in removed_prototypes:
                if idx != -1:
                    remain_prototypes.remove(idx)
                # Fit model
                knn.fit(X_train[remain_prototypes], y_train[remain_prototypes])
                # score test
                accuracy.append(knn.score(X_test, y_test))

            results["#Removed Prototypes"] = removed_prototypes
            results["total_distortions"].append(result["total_distortions"])
            results["reduction_rate"] = result["reduction_rate"]
            results["accuracy"].append(accuracy)

            min_len = min(min_len, len(result["total_distortions"]))

        total_distortions = [  # make all lists the same length
            obj_fun[:min_len] for obj_fun in results["total_distortions"]
        ]
        accuracy = [acc[:min_len] for acc in results["accuracy"]]
        removed_prototypes = results["#Removed Prototypes"][:min_len]
        reduction_rate = results["reduction_rate"][:min_len]

        total_distortion = np.mean(total_distortions, axis=0)
        accuracy = np.mean(accuracy, axis=0)

        total_distortion = [round(score, 2) for score in total_distortion]
        accuracy = [f"{acc:.2%}" for acc in accuracy]
        reduction_rate = [f"{rate:.2%}" for rate in reduction_rate]

        excel_content["Prototype Selection " + dataset] = {
            "#Removed": range(len(removed_prototypes)),
            "#Removed Prototypes": removed_prototypes,
            "Total Distortion": total_distortion,
            "Accuracy": accuracy,
            "Reduction Rate": reduction_rate,
        }
        pre_val = excel_total.get(
            "Prototype Selection " + dataset, {"#Removed": list(range(len(X)))}
        )
        pre_val[f"Accuracy {stability}"] = accuracy
        excel_total["Prototype Selection " + dataset] = pre_val
    save_to_excel(
        excel_content,
        f"prototype_selection acc test stability={stability} tmp",
        "horizontal",
    )
    logger.info("Results are saved to excel.")

    for key, value in excel_total.items():
        if value.get("Reduction Rate") is not None:
            value.pop("Reduction Rate")
        min_len = min([len(val) for val in value.values()])
        for key2, value2 in value.items():
            value[key2] = value2[:min_len]
        # Add reductaion rate
        value["Reduction Rate"] = [
            f"{(removed/(min_len+stability-1)):.2%}" for removed in range(min_len)
        ]

    save_to_excel(
        excel_total, f"prototype_selection acc test total {stability} tmp", "horizontal"
    )


if __name__ == "__main__":
    for stability in tqdm(stability_list, desc="stability progress", leave=False):
        run_stability(stability)
