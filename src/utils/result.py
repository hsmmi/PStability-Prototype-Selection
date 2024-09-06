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
        kappa: float,
        reduction: float,
        distortion: float,
        total_distortion: float,
        size: float,
        time: float,
    ):
        self.accuracy = accuracy
        self.kappa = kappa
        self.reduction = reduction
        self.distortion = distortion
        self.total_distortion = total_distortion
        self.size = size
        self.time = time

    def to_json(self):
        return {
            "Accuracy": self.accuracy,
            "Kappa": self.kappa,
            "Reduction": self.reduction,
            "Acc. * Red.": self.accuracy * self.reduction,
            "Kap. * Red.": self.kappa * self.reduction,
            "Distortion": self.distortion,
            "Total Distortion": self.total_distortion,
            "Size": self.size,
            "Time": self.time,
        }

    def format(self):
        return {
            "Accuracy": f"{self.accuracy:.2%}",
            "Kappa": f"{self.kappa:.4f}",
            "Reduction": f"{self.reduction:.2%}",
            "Acc. * Red.": f"{self.accuracy * self.reduction:.4f}",
            "Kap. * Red.": f"{self.kappa * self.reduction:.4f}",
            "Distortion": f"{self.distortion:.3f}",
            "Total Distortion": f"{self.total_distortion:.3f}",
            "Size": f"{self.size:.2f}",
            "Time": f"{self.time:.4f}",
        }


class AlgorithmResult:
    def __init__(
        self,
        algorithm: str = None,
    ):
        self.algorithm = algorithm
        self.n_runs = 0
        self.result = RunResult(0, 0, 0, 0, 0, 0, 0)

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

    def load_dict(self, data: dict):
        self.algorithm = data["algorithm"]
        self.n_runs = data["n_runs"]
        self.result = RunResult(
            data["result"]["Accuracy"],
            data["result"]["Kappa"],
            data["result"]["Reduction"],
            data["result"]["Distortion"],
            data["result"]["Total Distortion"],
            data["result"]["Size"],
            data["result"]["Time"],
        )

        return self


class DatasetResult:
    def __init__(self, dataset: str = None, n_folds: int = 5, k: int = 1):
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
            "Kappa",
            "Reduction",
            "Effectiveness",
            "Distortion",
            "Total Distortion",
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

        metrics = list(tmp_content[list(tmp_content.keys())[0]].keys())

        return {
            **{
                metric: [tmp_content[algorithm][metric] for algorithm in tmp_content]
                for metric in metrics
            },
        }

    def load_jsonl(self, filename, line=-1):
        """
        Load the dataset result from a JSONL file just one line.

        Parameters:
        filename (str): Name of the JSONL file.
        line (int): Line number to load from the file. If -1, load the last line.

        Returns:
        None
        """
        data = load_lines_in_range_jsonl(filename, line, line)
        self.dataset = data["dataset"]
        self.n_folds = data["n_folds"]
        self.k = data["k"]
        for algorithm in data["results"]:
            result = AlgorithmResult().load_dict(data["results"][algorithm])
            self.results[algorithm] = result

        return self

    def save_jsonl(self, filename):
        """
        Save the dataset result to a JSONL file.

        Parameters:
        filename (str): Name of the JSONL file.

        Returns:
        None
        """
        save_jsonl(filename, self)

    def save_to_excel(self, filename):
        """
        Save the dataset result to an Excel file.

        Parameters:
        filename (str): Name of the Excel file.

        Returns:
        None
        """
        from src.utils.excel import save_to_excel

        save_to_excel({self.dataset: self.ecxel_content()}, filename)


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


def save_jsonl(file_name, data: DatasetResult | dict):
    if file_name[-6:] == ".jsonl":
        file_name = file_name[:-6]
    file_name = LOG_PATH + file_name + ".jsonl"
    check_directory(file_name)
    if type(data) == DatasetResult:
        with open(file_name, "a") as f:
            f.write(json.dumps(data.to_json(), cls=NumpyEncoder) + "\n")
    else:
        with open(file_name, "a") as f:
            f.write(json.dumps(data, cls=NumpyEncoder) + "\n")


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
