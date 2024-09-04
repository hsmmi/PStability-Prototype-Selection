from src.utils.data_preprocessing import load_data
from src.utils.excel import save_to_excel


class Dataset:
    def __init__(self):
        self.name = None
        self.prototypes = None
        self.attributes = None
        self.classes = None

    def __repr__(self) -> str:
        return (
            f"Dataset: {self.name}\n"
            f"\tPrototypes: {self.prototypes}\n"
            f"\tAttributes: {self.attributes}\n"
            f"\tClasses: {self.classes}\n"
        )

    def load_dataset_by_name(self, dataset_name):
        self.name = dataset_name.replace("_", " ").capitalize()
        X, y = load_data(dataset_name)
        self.prototypes = len(X)
        self.attributes = len(X[0])
        self.classes = len(set(y))


class MyDataset:
    def __init__(self):
        self.datesets = []

    def __repr__(self) -> str:
        return "MyDataset" + f"\n{'='*30}\n".join(map(str, self.datesets))

    def add_dataset(self, dataset):
        self.datesets.append(dataset)


if __name__ == "__main__":
    datasets = [
        "appendicitis",  # ***
        "bupa",
        "circles_0.05_150",
        "ecoli",
        "glass",
        "haberman",
        "heart",
        "ionosphere",
        "iris",
        "liver",
        "moons_0.15_150",
        "movement_libras",
        "promoters",
        "sonar",
        "wine",
        "zoo",
    ]

    datasets.sort()

    my_dataset = MyDataset()

    for dataset_name in datasets:
        dataset = Dataset()
        dataset.load_dataset_by_name(dataset_name)
        my_dataset.add_dataset(dataset)

    print(my_dataset)

    excel_content = {}
    excel_content["Datasets"] = {
        "No.": [i for i in range(1, len(my_dataset.datesets) + 1)],
        "Dataset": [dataset.name for dataset in my_dataset.datesets],
        "Prototypes": [dataset.prototypes for dataset in my_dataset.datesets],
        "Attributes": [dataset.attributes for dataset in my_dataset.datesets],
        "Classes": [dataset.classes for dataset in my_dataset.datesets],
    }

    save_to_excel(excel_content, "datasets")
