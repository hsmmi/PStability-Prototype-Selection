import json
import numpy as np
import tabulate
from config import LOG_PATH
from config.log import get_logger
from src.utils.path import check_directory

logger = get_logger("result")


class RunResult:
    def __init__(
        self,
        accuracy: float,
        reduction: float,
        distortion: float,
        objective_function: float,
        size: float,
        time: float,
    ):
        self.accuracy = accuracy
        self.reduction = reduction
        self.effectiveness = self.accuracy * self.reduction
        self.distortion = distortion
        self.objective_function = objective_function
        self.size = size
        self.time = time

    def to_json(self):
        return {
            "accuracy": self.accuracy,
            "reduction": self.reduction,
            "effectiveness": self.effectiveness,
            "distortion": self.distortion,
            "objective_function": self.objective_function,
            "size": self.size,
            "time": self.time,
        }

    def format(self):
        return {
            "Accuracy": f"{self.accuracy:.2%}",
            "Reduction": f"{self.reduction:.2%}",
            "Effectiveness": f"{self.effectiveness:.2%}",
            "Distortion": f"{self.distortion:.3f}",
            "Objective Function": f"{self.objective_function:.3f}",
            "Size": f"{self.size:.2f}",
            "Time": f"{self.time:.4f}",
        }


class AlgorithmResult:
    def __init__(
        self,
        algorithm: str,
    ):
        self.algorithm = algorithm
        self.n_runs = 0
        self.result = RunResult(0, 0, 0, 0, 0, 0)

    def add_result(self, result: RunResult):
        for key in result.__dict__:
            setattr(
                self.result,
                key,
                (getattr(self.result, key) * self.n_runs + getattr(result, key))
                / (self.n_runs + 1),
            )

        self.n_runs += 1

    def to_json(self):
        return {
            "algorithm": self.algorithm,
            "n_runs": self.n_runs,
            "result": self.result.to_json(),
        }

    def format_dict(self):
        return {
            "Algorithm": self.algorithm,
            **self.result.format(),
        }

    def format_list(self):
        return [
            self.algorithm,
            *self.result.format().values(),
        ]


class DatasetResult:
    def __init__(self, dataset: str, n_folds: int = 5, k: int = 1):
        self.dataset = dataset
        self.n_folds = n_folds
        self.k = k
        self.results: dict[str, AlgorithmResult] = {}

    def add_result(self, result: AlgorithmResult):
        self.results[result.algorithm] = result

    def to_json(self):
        return {
            "dataset": self.dataset,
            "n_folds": self.n_folds,
            "k": self.k,
            "results": {
                algorithm: value.to_json() for algorithm, value in self.results.items()
            },
        }

    def print_tabulated(self):
        from tabulate import tabulate

        table = []
        for algorithm in self.results:
            table.append(self.results[algorithm].format_list())

        headers = [
            "Algorithm",
            "Accuracy",
            "Reduction",
            "Effectiveness",
            "Distortion",
            "Objective Function",
            "Size",
            "Time",
        ]

        # Add padding to the headers :^10
        headers = [f"{header:^10}" for header in headers]

        print(
            tabulate(
                table,
                headers=headers,
                tablefmt="fancy_grid",
                numalign="center",
                stralign="center",
            )
        )

    def ecxel_content(self):
        tmp_content = {}
        for algorithm in self.results:
            tmp_content[algorithm] = self.results[algorithm].format_dict()

        return {
            "Algorithms": list(tmp_content.keys()),
            "Accuracy": [
                tmp_content[algorithm]["Accuracy"] for algorithm in tmp_content
            ],
            "Reduction": [
                tmp_content[algorithm]["Reduction"] for algorithm in tmp_content
            ],
            "Effectiveness": [
                tmp_content[algorithm]["Effectiveness"] for algorithm in tmp_content
            ],
            "Distortion": [
                tmp_content[algorithm]["Distortion"] for algorithm in tmp_content
            ],
            "Objective Function": [
                tmp_content[algorithm]["Objective Function"]
                for algorithm in tmp_content
            ],
            "Size": [tmp_content[algorithm]["Size"] for algorithm in tmp_content],
            "Time": [tmp_content[algorithm]["Time"] for algorithm in tmp_content],
        }


def format_dict_results(results):
    return {
        key: {
            "Acc. Train": f"{results[key]['Acc. Train']/100:.4f}",
            "Acc. Test": f"{results[key]['Acc. Test']/100:.4f}",
            "Size": f"{results[key]['Size']/100:.4f}",
            "Distortion": f"{results[key]['Distortion']/100:.4f}",
            "Objective Function": f"{results[key]['Objective Function']/100:.4f}",
            "Reduction": f"{results[key]['Reduction']/100:.4f}",
            "Acc*Red": f"{results[key]['Acc*Red']/100:.4f}",
            "Time": f"{results[key]['Time']:.4f}",
        }
        for key in results
    }


def load_lines_in_range(file_name, start=None, end=None):
    lines = []

    with open(LOG_PATH + file_name, "r") as file:
        total_lines = sum(1 for _ in file)  # Calculate total number of lines
        file.seek(0)  # Reset file pointer to the beginning

        # Default start to 0 if None
        if start is None:
            start = 0
        elif start < 0:
            start = total_lines + start

        # Default end to last line if None
        if end is None:
            end = total_lines - 1
        elif end < 0:
            end = total_lines + end

        for current_line, line_content in enumerate(file):
            if start <= current_line <= end:
                lines.append(line_content.strip())
            if current_line > end:
                break

    # If there's only one line, return it as a string
    if len(lines) == 1:
        single_line = lines[0]
        return single_line

    return lines


def load_lines_in_range_jsonl(file_name, start=None, end=None):
    if file_name[-6:] == ".jsonl":
        file_name = file_name[:-6]

    lines = load_lines_in_range(file_name + ".jsonl", start, end)

    # If there's only one line, return it as a string
    if type(lines) == str:
        single_line = lines
        try:
            return json.loads(single_line)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON: {single_line}")
            return single_line  # Return as string if JSON parsing fails

    # If JSON parsing is requested and there are multiple lines
    try:
        return json.loads("\n".join(lines))
    except json.JSONDecodeError:
        return lines  # Return the list of lines if JSON parsing fails


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_jsonl(file_name, data: DatasetResult):
    if file_name[-6:] == ".jsonl":
        file_name = file_name[:-6]
    file_name = LOG_PATH + file_name + ".jsonl"
    check_directory(file_name)
    with open(file_name, "a") as f:
        f.write(json.dumps(data.to_json(), cls=NumpyEncoder) + "\n")


def log_result(
    result: DatasetResult,
    file_name: str,
    log_file: bool = True,
    log_console: bool = True,
) -> None:
    """
    Log the results of the experiment to a file and/or console.

        Parameters
        ----------
        result : dict
            Dictionary containing the results for each algorithm.
        file_name : str
            Name of the file to log the results.
        log_file : bool, optional
            Whether to log the results to a file (default is True).
        log_console : bool, optional
            Whether to log the results to the console (default is True).
    """

    if log_file:
        save_jsonl(file_name, result)

    if log_console:
        # Print in tabulated format
        result.print_tabulated()
