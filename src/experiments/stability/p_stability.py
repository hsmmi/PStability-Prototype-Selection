from src.algorithms.stability.p_stability import PStability
from src.utils.timer import measure_time
from config.log import get_logger

logger = get_logger("mylogger")

if __name__ == "__main__":
    DATASET = "iris_0_1"
    from src.utils.data_preprocessing import load_data

    X, y = load_data(DATASET)

    p_stability = PStability()
    with measure_time("fitting"):
        p_stability.fit(X, y)

    # with measure_time("Runtime: Exact missclassifications for each #p"):
    #     list_max_misses = p_stability.run_exact_miss(3)
    #     logger.info(f"Maximum misclassifications: {list_max_misses}")

    # with measure_time("Runtime: Exact p for each #missclassification"):
    #     list_exact_p = p_stability.run_exact_p(2)
    #     logger.info(f"Maximum p for stability: {list_exact_p}")

    # associated_p = p_stability.convert_misses_to_p_list(list_exact_p)
    # logger.info(f"{[(p, misses) for p, misses in enumerate(associated_p)]}")

    # list_exact_misses = p_stability.convert_misses_to_p_list(list_exact_p)
    # logger.info(f"{[(p, misses) for p, misses in enumerate(list_exact_misses)]}")

    # with measure_time("Runtime: Lower bound p for each #missclassification"):
    #     list_lower_bound_p = p_stability.run_lower_bound_p(20)
    #     logger.info(f"Lower bound p for stability: {list_lower_bound_p}")

    # with measure_time("Runtime: Upper bound p for each #missclassification"):
    #     list_upper_bound_p = p_stability.run_upper_bound_p(20)
    #     logger.info(f"Upper bound p for stability: {list_upper_bound_p}")

    with measure_time("Runtime: Fuzzy missclassifications for each #p"):
        list_fuzzy_misses = p_stability.run_fuzzy_missclassification(10)
        logger.info(f"Fuzzy misclassifications: {list_fuzzy_misses}")
