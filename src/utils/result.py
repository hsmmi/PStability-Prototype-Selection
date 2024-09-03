import json
import numpy as np
import tabulate
from config import LOG_PATH
from config.log import get_logger

logger = get_logger("result")


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


def save_jsonl(file_name, data):
    with open(LOG_PATH + file_name + ".jsonl", "a") as f:
        f.write(json.dumps(data, cls=NumpyEncoder) + "\n")


def log_result(
    result: dict,
    file_name: str,
    dataset: str,
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
        dataset : str
            Name of the dataset used in the experiment.
        log_file : bool, optional
            Whether to log the results to a file (default is True).
        log_console : bool, optional
            Whether to log the results to the console (default is True).
    """
    formatted_result = {
        key: {
            "Acc. Train": round(result[key][0] * 100, 2),
            "Acc. Test": round(result[key][1] * 100, 2),
            "Size": round(result[key][2], 2),
            "Distortion": round(result[key][3], 2),
            "Objective Function": round(result[key][4], 2),
            "Reduction": round(result[key][5] * 100, 2),
            "Time": round(result[key][6], 3),
        }
        for key in result
    }

    if log_file:
        save_jsonl(file_name, {"dataset": dataset, "results": formatted_result})

    if log_console:
        # Print in tabulated format
        table = []
        for key in result:
            table.append(
                [
                    key,
                    f"{result[key][0]:.2%}",
                    f"{result[key][1]:.2%}",
                    f"{result[key][2]:.2f}",
                    f"{result[key][3]:.2f}",
                    f"{result[key][4]:.2f}",
                    f"{result[key][5]:.2%}",
                    f"{result[key][6]:.3f}s",
                ]
            )

        headers = [
            "Algorithm",
            "Acc. Train",
            "Acc. Test",
            "Size",
            "Distortion",
            "Objective Function",
            "Reduction",
            "Time",
        ]

        # Add padding to the headers :^10
        headers = [f"{header:^10}" for header in headers]

        print(
            tabulate.tabulate(
                table,
                headers,
                tablefmt="fancy_grid",
                numalign="center",
                stralign="center",
            )
        )
