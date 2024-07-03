import random
import time
from src.algorithms.cnn import condensed_nearest_neighbor
from src.utils.data_preprocessing import load_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and preprocess the data
data_path = "data/raw/data.csv"
# print pwd

df = load_data(data_path)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# set random seed to 42
random.seed(42)

# Train the KNN classifier on the reduced dataset
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluate the classifier
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Start timer
start_time = time.time()

# Apply CNN algorithm
X_reduced, y_reduced = condensed_nearest_neighbor(X_train, y_train)

# End timer
end_time = time.time()

# Train the KNN classifier on the reduced dataset
knn_reduced = KNeighborsClassifier(n_neighbors=3)
knn_reduced.fit(X_reduced, y_reduced)

# Evaluate the classifier
y_pred_reduced = knn_reduced.predict(X_test)
accuracy_reduced = accuracy_score(y_test, y_pred_reduced)


# Log the results
log_path = "results/logs/experiment_1.log"
with open(log_path, "w") as log_file:
    log_file.write(f"Original accuracy: {accuracy}\n")
    log_file.write(f"Reduced accuracy: {accuracy_reduced}\n")
    log_file.write(f"Original size: {len(X_train)}\n")
    log_file.write(f"Reduced size: {len(X_reduced)}\n")
    log_file.write(
        f"Reduction percentage: {100 * (1 - len(X_reduced) / len(X_train)):.2f}%\n"
    )
    log_file.write(f"Execution time: {end_time - start_time:.2f} seconds\n")
    log_file.write("\n")

print(f"Original accuracy: {accuracy}")
print(f"Reduced accuracy: {accuracy_reduced}")
print(f"Original size: {len(X_train)}")
print(f"Reduced size: {len(X_reduced)}")
print(f"Reduction percentage: {100 * (1 - len(X_reduced) / len(X_train)):.2f}%")
print(f"Execution time: {end_time - start_time:.2f} seconds")
