import pandas as pd
import logging
from typing import Optional

class DataLoader:
    def __init__(self, file_path: str, header: Optional[int] = 'infer'):
        """
        Initializes the DataLoader with the path to the CSV file and the header configuration.

        Args:
            file_path (str): The path to the CSV file.
            header (Optional[int, list of int, or None], optional): Row number(s) to use as the column names, and the start of the data. 
                                                                   Default is 'infer', which means the first row is used as column names.
                                                                   If None, no header is used.
        """
        self.file_path = file_path
        self.header = header

    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Loads data from the CSV file into a pandas DataFrame.

        Returns:
            Optional[pd.DataFrame]: The loaded DataFrame, or None if an error occurs.
        """
        try:
            if self.header is None:
                df = pd.read_csv(self.file_path, header=self.header)
            else:
                df = pd.read_csv(self.file_path)
            logging.info(f"Data loaded successfully from {self.file_path}")
            return df
        except Exception as e:
            logging.error(f"Error loading data from {self.file_path}: {e}")
            return None