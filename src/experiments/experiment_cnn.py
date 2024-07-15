import random
from src.algorithms.cnn import condensed_nearest_neighbor
from src.utils.data_preprocessing import load_data
from src.utils.evaluation_metrics import compare_prototype_selection
from src.utils.result import log_result

# set random seed to 42
random.seed(42)

# Log the results
log_path = "results/logs/experiment_cnn.log"

# Load and preprocess the data
data_path = "data/raw/data.csv"

df = load_data(data_path)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

result = compare_prototype_selection(X, y, [condensed_nearest_neighbor])

log_result(result, log_path, "data")
