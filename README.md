# Diabetes Readmission Prediction

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
The project is organized into several directories:

- **analysis**: Contains Jupyter notebooks for exploratory data analysis. The `analyze_src` subdirectory includes scripts for basic data inspection.
- **data**: Includes the dataset file (`diabetes+130-us+hospitals+for+years+1999-2008.zip`) and scripts for data loading and preprocessing.
- **extracted_data**: Contains extracted CSV files (`diabetic_data.csv` and `IDS_mapping.csv`) from the dataset.
- **models**: Contains scripts for model training and prediction.
- **src**: Includes the main script for data ingestion.

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
