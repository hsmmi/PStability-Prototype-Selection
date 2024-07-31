from pathlib import Path
import os
import logging


class ProjectPath:
    def __init__(self, current_file_path: str, marker: str = ".git"):
        """
        Initialize the ProjectPath object.

        Args:
            current_file_path (str): The path to the current file using this class.
            marker (str): The marker file or directory to identify the project root.
                          Defaults to '.git'.
        """
        self.current_file_path = Path(current_file_path).resolve()
        self.marker = marker
        self.project_root = self.find_project_root()

    def find_project_root(self) -> Path:
        """
        Find the project root directory by searching for a marker.

        Returns:
            Path: The path to the project root.

        Raises:
            FileNotFoundError: If the project root with the specified marker is not found.
        """
        for parent in self.current_file_path.parents:
            if (parent / self.marker).exists():
                return parent
        raise FileNotFoundError(f"Project root with marker '{self.marker}' not found.")

    def get_relative_path(self) -> str:
        """
        Get the relative path of the current file from the project root.

        Returns:
            str: The relative path from the project root to the current file.
        """
        return str(self.current_file_path.relative_to(self.project_root))

    def get_full_path(self) -> str:
        """
        Get the full path of the current file.

        Returns:
            str: The absolute path to the current file.
        """
        return str(self.current_file_path)

    def get_safe_filename(self) -> str:
        """
        Make the current file path safe for saving files by replacing directory separators
        with underscores and removing the file extension.

        Returns:
            str: The safe filename for saving files.
        """
        relative_path = self.get_relative_path()
        # Use os.path.sep instead of Path.sep
        safe_filename = relative_path.replace(os.path.sep, "_")
        return Path(safe_filename).stem


# Usage example:
if __name__ == "__main__":
    try:
        current_file_path = __file__
        project_path = ProjectPath(current_file_path)
        relative_file_path = project_path.get_relative_path()
        print(f"Relative path from project root: {relative_file_path}")
    except FileNotFoundError as e:
        logging.error(e)
