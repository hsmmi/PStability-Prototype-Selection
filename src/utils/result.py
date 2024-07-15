import json
import tabulate


def log_result(result, log_path, dataset, log_file=True, log_console=True):
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
        with open(log_path, "a") as f:
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
