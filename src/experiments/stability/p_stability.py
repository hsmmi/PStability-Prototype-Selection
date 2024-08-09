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

    with measure_time("running"):
        list_max_misses = p_stability.run_misses(range(1, 4))
        logger.info(f"Maximum misclassifications: {list_max_misses}")

    with measure_time("relaxed running"):
        list_max_misses = p_stability.run_relaxed_misses(range(1, 30))
        logger.info(f"Maximum misclassifications (relaxed): {list_max_misses}")

    with measure_time("Find maximum p for stability"):
        max_p = p_stability.run_max_p(range(1, 4))
        logger.info(f"Maximum p for stability: {max_p}")
