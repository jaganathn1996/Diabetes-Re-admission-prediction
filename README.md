# Diabetes Readmission Prediction

## Key Files

- **diabetes_readmission_predection.ipynb**: Main Jupyter notebook for the project.
- **requirements.txt**: List of dependencies.
- analysis/analyze_src/**basic_data_inspection.py**: Script for basic data inspection.
- data/**data_loader.py**: Script for loading data.
- data/**preprocessing.py**: Script for preprocessing data.
- models/**model_trainer.py**: Script for training models.
- src/**ingest_data.py**: Script for ingesting data.

## Introduction
This project analyzes a dataset containing diabetes-related information from 130 US hospitals over 10 years (1999-2008). The goal is to uncover insights and trends to improve diabetes care and management.

## Dataset Description
The dataset includes:
- **Patient ID**: Unique identifier for each patient
- **Admission Type**: Type of hospital admission (e.g., emergency, urgent, elective)
- **Diagnosis**: Primary and secondary diagnoses
- **Treatment**: Medications and procedures administered
- **Outcome**: Patient outcomes (e.g., discharge status, readmission)
- **Demographics**: Age, gender, race, etc.
- **Hospital Information**: Hospital ID, location, etc.

## Objectives
1. Identify trends and patterns in diabetes treatment and outcomes.
2. Analyze the impact of demographic factors on diabetes management.
3. Evaluate the effectiveness of different treatment protocols.
4. Identify potential areas for improvement in diabetes care.

## Project Structure

The project is organized into several directories and important files:

- **analysis**: 
  - `analyze_src`: Includes scripts for basic data inspection.
  
- **data**: 
  - `data_loader.py`: Script for loading data.
  - `diabetes+130-us+hospitals+for+years+1999-2008.zip`: The main dataset file.
  - `preprocessing.py`: Script for preprocessing data.
  
- **extracted_data**: 
  - `diabetic_data.csv`: Extracted data from the dataset.
  - `IDS_mapping.csv`: Mapping of IDs from the dataset.
  
- **models**: 
  - `model_trainer.py`: Script for training models.
  
- **src**: 
  - `ingest_data.py`: Main script for data ingestion.

- **Root Directory**:
  - `diabetes_readmission_predection.ipynb`: **Main Jupyter notebook** for the project, containing the core analysis and predictions.
  - `Diabetes_Readmission_Summary.docx`: Summary document of the project findings.
  - `README.md`: This file, providing an overview of the project structure.
  - `requirements.txt`: Lists the dependencies required for the project.

## Installation
Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage
### Data Loading and Exploration
Use the `DataIngestorFactory` class from the `src/ingest_data.py` module to perform data ingestion and initial exploration.

### Data Inspection
Inspect data types and summary statistics using functions in `analysis/analyze_src/basic_data_inspection.py`.

### Handling Missing Values
Summarize and handle missing values as outlined in the analysis notebook.

### Data Pre-processing
Preprocess data using mappings and groupings in `data/preprocessing.py`.

### Model Training and Prediction
Train and predict using different scenarios in `models/model_trainer.py`.

### Conclusion
Compare model performance on cleaned and uncleaned datasets, balancing accuracy and computational efficiency.
