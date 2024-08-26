import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from src.algorithms.stability.my_prototype_selection import PrototypeSelection
from src.utils.excel import save_to_excel
from config.log import get_logger

logger = get_logger("mylogger")
logger.setLevel("WARNING")

n_folds = 5

if __name__ == "__main__":
    dataset_list = [
        "circles_0.05_undersampled",
        "moons_0.15_undersampled",
        "iris_undersampled",
        "wine_undersampled",
        "iris_0_1",
        "iris_0_2",
        "iris_1_2",
        "iris",
        "wine",
    ]

    from src.utils.data_preprocessing import load_data

    excel_content = {}

    prototype_selection = PrototypeSelection()

    for DATASET in tqdm(dataset_list, desc="Datasets progress", leave=False):
        X, y = load_data(DATASET)

        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        results = {"objective_functions": [], "accuracy": []}
        min_len = len(X)
        for train_index, test_index in tqdm(
            kf.split(X, y), total=n_folds, desc="K-Fold progress", leave=False
        ):
            X_train, y_train = X[train_index], y[train_index]

            prototype_selection.fit(X_train, y_train)

            result = prototype_selection.prototype_reduction(5)

            removed_prototypes = result["removed_prototypes"]
            base_objective_function = result["base_objective_function"]
            idx_min_objective_function = result["idx_min_objective_function"]
            last_idx_under_base = result["last_idx_under_base"]

            results["#Removed Prototypes"] = removed_prototypes
            results["objective_functions"].append(result["objective_functions"])
            results["accuracy"].append(result["accuracy"])
            results["reduction_rate"] = result["reduction_rate"]

            min_len = min(min_len, len(result["objective_functions"]))

        objective_functions = [  # make all lists the same length
            objective_functions[:min_len]
            for objective_functions in results["objective_functions"]
        ]
        accuracies = [accuracy[:min_len] for accuracy in results["accuracy"]]
        removed_prototypes = results["#Removed Prototypes"][:min_len]
        reduction_rate = results["reduction_rate"][:min_len]

        objective_function = np.mean(objective_functions, axis=0)
        accuracy = np.mean(accuracies, axis=0)

        objective_function = [round(score, 2) for score in objective_function]
        accuracy = [round(acc * 100, 2) for acc in accuracy]
        reduction_rate = [round(rate * 100, 2) for rate in reduction_rate]

        excel_content["Prototype Selection " + DATASET] = {
            "#Removed": range(len(removed_prototypes)),
            "#Removed Prototypes": removed_prototypes,
            "Total Scores": objective_function,
            "Accuracy": accuracy,
            "Reduction Rate": reduction_rate,
        }
        print(excel_content["Prototype Selection " + DATASET])
    save_to_excel(excel_content, "prototype_selection", "horizontal")
    logger.info("Results are saved to excel.")
