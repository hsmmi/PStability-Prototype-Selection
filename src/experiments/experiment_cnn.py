import random
from src.algorithms.cnn import condensed_nearest_neighbor
from src.utils.data_preprocessing import load_data
from src.utils.evaluation_metrics import compare_prototype_selection

# set random seed to 42
random.seed(42)

# Load and preprocess the data
data_path = "data/raw/data.csv"

df = load_data(data_path)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

result = compare_prototype_selection(X, y, [condensed_nearest_neighbor])

# Log the results
log_path = "results/logs/experiment_cnn.log"

# print(f"Original accuracy: {accuracy*100:.2f}%")
# print(f"Reduced accuracy: {accuracy_reduced*100:.2f}%")
# print(f"Original size: {len(X_train)}")
# print(f"Reduced size: {len(X_reduced)}")
# print(f"Reduction percentage: {100 * (1 - len(X_reduced) / len(X_train)):.2f}%")
# print(f"Execution time: {end_time - start_time:.2f} seconds")

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
