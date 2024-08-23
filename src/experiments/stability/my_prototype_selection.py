from sklearn.neighbors import KNeighborsClassifier
from src.algorithms.stability.my_prototype_selection import PrototypeSelection
from src.utils.timer import measure_time
from src.utils.excel import save_to_excel
from config.log import get_logger
from src.utils.visualization import plot_algorithm_results

logger = get_logger("mylogger")

if __name__ == "__main__":
    dataset_list = [
        "iris_0_1",
        "iris_0_2",
        "iris_1_2" "circles_0.05_undersampled",
        "moons_0.15_undersampled",
    ]

    from src.utils.data_preprocessing import load_data

    excel_content = {}

    for DATASET in dataset_list:
        X, y = load_data(DATASET)

        prototype_selection = PrototypeSelection()

        # K-Fold

        prototype_selection.fit(X, y)

        with measure_time("Runtime: Prototype Selection for " + DATASET):
            result = prototype_selection.prototype_reduction(5)

            removed_prototypes = result["removed_prototypes"]
            total_scores = [round(score, 2) for score in result["total_scores"]]
            accuracy = [f"{acc:.2%}" for acc in result["accuracy"]]
            base_total_score = result["base_total_score"]
            idx_min_total_score = result["idx_min_total_score"]
            last_idx_under_base = result["last_idx_under_base"]

            logger.debug(
                f"\nRemoved prototypes:\n{removed_prototypes}\n\n"
                f"Total Scores:\n{total_scores}\n\n"
                f"Accuracy:\n{accuracy}\n\n"
                f"Base Total Score: {base_total_score}\n\n"
                f"Index of Min Total Score: {idx_min_total_score}\n\n"
                f"Last Index Under Base: {last_idx_under_base}"
            )
            excel_content["Prototype Selection " + DATASET] = {
                "#Removed": range(len(removed_prototypes)),
                "#Removed Prototypes": removed_prototypes,
                "Total Scores": total_scores,
                "Accuracy": accuracy,
            }
    save_to_excel(excel_content, "prototype_selection", "horizontal")
    logger.info("Results are saved to excel.")
