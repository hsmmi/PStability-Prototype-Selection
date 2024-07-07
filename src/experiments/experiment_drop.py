import random

from sklearn.datasets import make_moons
from src.utils.evaluation_metrics import compare_prototype_selection
from src.algorithms.drop import DROP as DROP

# set random seed to 42
random.seed(42)

X, y = make_moons(n_samples=500, noise=0.5, random_state=42)

result = compare_prototype_selection(
    X, y, [DROP(3, "DROP1").fit, DROP(3, "DROP2").fit, DROP(3, "DROP3").fit]
)

# Log the results
log_path = "results/logs/experiment_drop.log"

with open(log_path, "a") as log_file:
    for key, value in result.items():
        log_file.write(f". {key}: {value}")
        log_file.write("\n")
    log_file.write("\n")

for key in result:
    print(f"{key} Algorithm Results")
    print(f"Accuracy: {result[key][0]*100:.2f}%")
    print(f"Size: {result[key][1]}")
    print(f"Reduction Percentage: {result[key][2]:.2f}%")
    print(f"Execution Time: {result[key][3]:.2f} seconds")
    print("\n")
