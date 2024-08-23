from src.algorithms.stability.p_stability import PStability
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

    with measure_time("Runtime: Exact p for each Stability"):
        list_stability = [0, 1, 2]
        list_exact_p = p_stability.run_exact_p(list_stability)
        logger.info(f"Exact p: {list_exact_p}")
        excel_content["Exact p"] = {
            "stability": list_stability,
            "p": list_exact_p,
        }

    with measure_time("Runtime: Lower bound p for each stability"):
        list_stability = list(range(21))
        list_lower_bound_p = p_stability.run_lower_bound_p(list_stability)
        logger.info(f"Lower bound p for stability: {list_lower_bound_p}")
        excel_content["Lower Bound p"] = {
            "stability": list_stability,
            "p": list_lower_bound_p,
        }

    with measure_time("Runtime: Better upper bound p for each stability"):
        list_stability = list(range(21))
        list_better_upper_bound_p = p_stability.run_better_upper_bound_p(list_stability)
        logger.info(f"Better upper bound p for stability: {list_better_upper_bound_p}")
        excel_content["Better Upper Bound p"] = {
            "stability": list_stability,
            "p": list_better_upper_bound_p,
        }

    with measure_time("Runtime: Upper bound p for each stability"):
        list_stability = list(range(21))
        list_upper_bound_p = p_stability.run_upper_bound_p(list_stability)
        logger.info(f"Upper bound p for stability: {list_upper_bound_p}")
        excel_content["Upper Bound p"] = {
            "stability": list_stability,
            "p": list_upper_bound_p,
        }

    with measure_time("Runtime: Exact Stability for each p"):
        list_p = [0, 1, 2]
        list_exact_stability = p_stability.run_exact_stability(list_p)
        logger.info(f"Exact Stability: {list_exact_stability}")
        excel_content["Exact MissclasStabilityifications"] = {
            "p": list_p,
            "stability": list_exact_stability,
        }

    with measure_time("Runtime: Lower bound stability for each p"):
        list_p = list(range(101))
        list_lower_bound_stability = p_stability.run_lower_bound_stability(list_p)
        logger.info(f"Lower bound stability for p: {list_lower_bound_stability}")
        excel_content["Lower Bound Stability"] = {
            "p": list_p,
            "stability": list_lower_bound_stability,
        }

    with measure_time("Runtime: Better lower bound stability for each p"):
        list_p = list(range(101))
        list_better_lower_bound_stability = (
            p_stability.run_better_lower_bound_stability(list_p)
        )
        logger.info(
            f"Better lower bound stability for p: {list_better_lower_bound_stability}"
        )
        excel_content["Better Lower Bound Stability"] = {
            "p": list_p,
            "stability": list_better_lower_bound_stability,
        }

    with measure_time("Runtime: Upper bound stability for each p"):
        list_p = list(range(101))
        list_upper_bound_stability = p_stability.run_upper_bound_stability(list_p)
        logger.info(f"Upper bound stability for p: {list_upper_bound_stability}")
        excel_content["Upper Bound Stability"] = {
            "p": list_p,
            "stability": list_upper_bound_stability,
        }

    with measure_time("Runtime: Crisped Stability for p"):
        list_p = list(range(101))
        list_crisped_stability = p_stability.run_crisped_stability(list_p)
        list_crisped_stability_score = [tuple[0] for tuple in list_crisped_stability]
        logger.info(
            f"Crisped stabilityclassification score: {list_crisped_stability_score}"
        )
        excel_content["Crisped Stability(lower)"] = {
            "p": list_p,
            "Crisped Miss": list_crisped_stability_score,
        }

    with measure_time("Runtime: Fuzzy Stability for p"):
        list_p = list(range(101))
        list_fuzzy_stability = p_stability.run_fuzzy_stability(list_p)
        list_fuzzy_stability_score = [
            round(tuple[0], 2) for tuple in list_fuzzy_stability
        ]
        logger.info(
            f"Fuzzy stabilityclassification score: {list_fuzzy_stability_score}"
        )
        excel_content["Fuzzy Stability(upper)"] = {
            "p": list_p,
            "Fuzzy Stability": list_fuzzy_stability_score,
        }

    # save_to_excel(excel_content, "p_stability", mode="horizontal")
