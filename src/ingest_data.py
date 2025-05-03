import os
import zipfile
from abc import ABC, abstractmethod
import json
import shutil

import pandas as pd

# Define an abstract class for Data Ingestor
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> dict:
        """Abstract method to ingest data from a given file."""
        pass

# Implement a concrete class for ZIP ingestion
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> dict:
        """Extracts a .zip file and returns metadata about the contents."""
        # Ensure the file is a .zip
        if not file_path.endswith(".zip"):
            raise ValueError("The provided file is not a .zip file.")

        # Remove the directory if it exists
        if os.path.exists("extracted_data"):
            shutil.rmtree("extracted_data")

        # Extract the zip file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("extracted_data")
        
        # Analyze the extracted files
        extracted_files = os.listdir("extracted_data")
        file_metadata = []

        for file_name in extracted_files:
            file_path = os.path.join("extracted_data", file_name)
            file_info = {
                "file_name": file_name,
                "file_type": os.path.splitext(file_name)[1],
                "file_path": os.path.abspath(file_path).replace("\\","/"),
            }
            if file_info["file_type"] == ".csv":
                df = pd.read_csv(file_path)
                file_info["num_rows"] = df.shape[0]
                file_info["num_columns"] = df.shape[1]
            file_metadata.append(file_info)

        # Return metadata about the extracted files
        return file_metadata

# Implement a Factory to create DataIngestors
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """Returns the appropriate DataIngestor based on file extension."""
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")

if __name__ == "__main__":

    # Specify the file path
    file_path = "C:/Users/u1109539/Downloads/CC_Project/diabetes-readmission-predictor/data/diabetes+130-us+hospitals+for+years+1999-2008.zip"

    # Determine the file extension
    file_extension = os.path.splitext(file_path)[1]

    # Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # Ingest the data and get metadata
    metadata = data_ingestor.ingest(file_path)

    # Print metadata
    print(json.dumps(metadata, indent=4))