from src.utils.result import load_lines_in_range_jsonl
from src.utils.visualization import plot_bounds
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
import matplotlib.pyplot as plt


result = load_lines_in_range_jsonl("p_stability", -1)
dataset = result["dataset"]
results = result["results"]

plot_bounds(results, dataset)
