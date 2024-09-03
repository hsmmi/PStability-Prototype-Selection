from src.utils.path import check_directory
import pandas as pd
from config import LOG_PATH


def save_to_excel(results: dict, file_name: str, mode: str = "horizontal") -> None:
    """
    Save a dictionary of results to an Excel file.

    The dictionary structure should be as follows:

        {
            "Table1_name": {
                "Column1_name": [1, 2, 3],
                "Column2_name": [4, 5, 6],
                ...
            },
            "Table2_name": {
                "Column1_name": [7, 8, 9],
                "Column2_name": [10, 11, 12],
                ...
            },
            ...
        }

    Parameters
    ----------
    results : dict
        Dictionary containing the results for each table.
    file_name : str
        Name of the file to save the results.
    mode : str, optional
        Mode to save the results: "new_sheet", "horizontal", or "vertical".
        Default is "horizontal".
    """

    file_name = LOG_PATH + file_name + ".xlsx"

    # Check if the directory exists, and create it if it does not
    check_directory(file_name)

    with pd.ExcelWriter(file_name, engine="xlsxwriter") as writer:
        if mode == "new_sheet":
            for table_name, table_data in results.items():
                df = pd.DataFrame(table_data)
                df.to_excel(writer, sheet_name=table_name, startrow=1, index=False)

                # Add borders around the table
                worksheet = writer.sheets[table_name]
                n_rows, n_cols = df.shape

                # Create formats for cells
                border_format = writer.book.add_format(
                    {"border": 1, "align": "center", "valign": "vcenter"}
                )

                title_format = writer.book.add_format(
                    {"bold": True, "align": "center", "valign": "vcenter", "border": 1}
                )

                # Calculate and set column widths based on the longest entry
                for col_num, col_name in enumerate(df.columns):
                    max_len = max(
                        df[col_name].astype(str).apply(len).max(), len(col_name)
                    )
                    worksheet.set_column(col_num, col_num, max_len + 2)

                # Calculate total width required for the title
                title_len = len(table_name)
                total_width = sum(
                    max(
                        df[col_name].astype(str).apply(len).max(),
                        len(col_name),
                    )
                    + 2
                    for col_name in df.columns
                )
                if total_width < title_len + 2:
                    adjusted_width = (title_len + 2) // n_cols + 1
                    worksheet.set_column(0, n_cols - 1, adjusted_width)

                # Merge and center the title across the table columns
                worksheet.merge_range(0, 0, 0, n_cols - 1, table_name, title_format)

                # Apply the border format around the table cells
                for row_num in range(1, n_rows + 2):  # +2 to account for header row
                    for col_num in range(n_cols):
                        worksheet.write(
                            row_num,
                            col_num,
                            (
                                df.iloc[row_num - 2, col_num]
                                if row_num > 1
                                else df.columns[col_num]
                            ),
                            border_format,
                        )

        else:
            # Use a single sheet
            worksheet = writer.book.add_worksheet("Results")
            start_row, start_col = 0, 0

            # Define formats
            title_format = writer.book.add_format(
                {"bold": True, "align": "center", "valign": "vcenter", "border": 1}
            )
            border_format = writer.book.add_format(
                {"border": 1, "align": "center", "valign": "vcenter"}
            )

            for table_name, table_data in results.items():
                df = pd.DataFrame(table_data)
                n_rows, n_cols = df.shape

                # Calculate and set column widths based on the longest entry
                for col_num, col_name in enumerate(df.columns):
                    max_len = max(
                        df[col_name].astype(str).apply(len).max(), len(col_name)
                    )
                    worksheet.set_column(
                        start_col + col_num, start_col + col_num, max_len + 2
                    )

                # Calculate total width required for the title
                title_len = len(table_name)
                total_width = sum(
                    max(
                        df[col_name].astype(str).apply(len).max(),
                        len(col_name),
                    )
                    + 2
                    for col_name in df.columns
                )
                if total_width < title_len + 2:
                    adjusted_width = (title_len + 2) // n_cols + 1
                    worksheet.set_column(
                        start_col, start_col + n_cols - 1, adjusted_width
                    )

                # Merge and write the table name as the title
                worksheet.merge_range(
                    start_row,
                    start_col,
                    start_row,
                    start_col + n_cols - 1,
                    table_name,
                    title_format,
                )

                # Apply the border format around the table cells
                for row_num in range(
                    start_row + 1, start_row + n_rows + 2
                ):  # +2 to account for header row
                    for col_num in range(start_col, start_col + n_cols):
                        if row_num == start_row + 1:
                            worksheet.write(
                                row_num,
                                col_num,
                                df.columns[col_num - start_col],
                                border_format,
                            )
                        else:
                            worksheet.write(
                                row_num,
                                col_num,
                                df.iloc[row_num - start_row - 2, col_num - start_col],
                                border_format,
                            )

                if mode == "horizontal":
                    # Move to the next column (add one empty column in between)
                    start_col += n_cols + 1
                elif mode == "vertical":
                    # Move to the next row (add one empty row in between)
                    start_row += n_rows + 3
