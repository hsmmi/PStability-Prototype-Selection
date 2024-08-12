from src.algorithms.stability.p_stability import PStability
from src.algorithms.stability.my_prototype_selection import PrototypeSelection
from src.utils.timer import measure_time
from config.log import get_logger
from src.utils.excel import save_to_excel

logger = get_logger("mylogger")

if __name__ == "__main__":
    DATASET = "iris_0_1"
    from src.utils.data_preprocessing import load_data

    X, y = load_data(DATASET)

    excel_content = {}

    p_stability = PStability()
    with measure_time("fitting"):
        p_stability.fit(X, y)

    with measure_time("Runtime: Exact misclassifications for each #p"):
        list_exact_misses = p_stability.run_exact_miss(2)
        logger.info(f"Maximum misclassifications: {list_exact_misses}")
        excel_content["Exact Missclassifications"] = {
            "#p": range(len(list_exact_misses)),
            "#misses": list_exact_misses,
        }

    with measure_time("Runtime: Exact p for each #missclassification"):
        list_exact_p = p_stability.run_exact_p(2)
        logger.info(f"Maximum p for stability: {list_exact_p}")
        excel_content["Exact p"] = {
            "#misses": range(len(list_exact_p)),
            "#p": list_exact_p,
        }

    list_dual_exact_p = p_stability.convert_misses_to_p_list(list_exact_p)
    logger.info(
        "List of exact missclassifications corresponding to p with conversion: ",
        [(p, misses) for p, misses in enumerate(list_dual_exact_p)],
    )
    excel_content["Dual Exact p"] = {
        "#p": range(len(list_dual_exact_p)),
        "#misses": list_dual_exact_p,
    }

    with measure_time("Runtime: Lower bound p for each #missclassification"):
        list_lower_bound_p = p_stability.run_lower_bound_p(20)
        logger.info(f"Lower bound p for stability: {list_lower_bound_p}")
        excel_content["Lower Bound p"] = {
            "#misses": range(len(list_lower_bound_p)),
            "#p": list_lower_bound_p,
        }

    with measure_time("Runtime: Upper bound p for each #missclassification"):
        list_upper_bound_p = p_stability.run_upper_bound_p(20)
        logger.info(f"Upper bound p for stability: {list_upper_bound_p}")
        excel_content["Upper Bound p"] = {
            "#misses": range(len(list_upper_bound_p)),
            "#p": list_upper_bound_p,
        }

    with measure_time("Runtime: Fuzzy misclassifications for each #p"):
        list_fuzzy_misses = p_stability.run_fuzzy_missclassification(10)
        logger.info(f"Fuzzy misclassifications: {list_fuzzy_misses}")
        excel_content["Fuzzy Missclassifications"] = {
            "#p": range(len(list_fuzzy_misses)),
            "#fuzzy_misses": list_fuzzy_misses,
        }

    prototype_selection = PrototypeSelection()
    prototype_selection.fit(X, y)

    with measure_time("Runtime: Prototype Selection"):
        removed_prototypes, _ = prototype_selection.prototype_reduction(10)
        logger.info(f"Removed prototypes: {removed_prototypes}")
        excel_content["Prototype Selection"] = {
            "#prototypes": range(len(removed_prototypes)),
            "#removed_prototypes": removed_prototypes,
        }

    save_to_excel(excel_content, "p_stability", mode="horizontal")
