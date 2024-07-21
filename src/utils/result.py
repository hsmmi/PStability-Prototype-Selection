import json
import tabulate
from config import LOG_PATH


def log_result(result, file_name, dataset, log_file=True, log_console=True) -> None:
    """
    Log the results of the experiment to a file and/or console.

    Parameters:
    result (dict): Dictionary containing the results for each algorithm.
    file_name (str): Name of the file to log the results.
    dataset (str): Name of the dataset used in the experiment.
    log_file (bool): Whether to log the results to a file.
    log_console (bool): Whether to log the results to the console
    """
    formatted_result = {
        key: {
            "Accuracy": round(result[key][0] * 100, 2),
            "Size": result[key][1],
            "Reduction": round(result[key][2] * 100, 2),
            "Time": round(result[key][3], 3),
        }
        for key in result
    }

    if log_file:
        with open(LOG_PATH + file_name + ".log", "a") as f:
            f.write(
                json.dumps({"dataset": dataset, "results": formatted_result}) + "\n"
            )

    if log_console:
        # Print in tabulated format
        table = []
        for key in result:
            table.append(
                [
                    key,
                    f"{result[key][0]*100:.2f}%",
                    result[key][1],
                    f"{result[key][2]*100:.2f}%",
                    f"{result[key][3]:.3f}s",
                ]
            )

        headers = [
            "Algorithm",
            "Accuracy",
            "Size",
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
                numalign="right",
                stralign="right",
            )
        )
