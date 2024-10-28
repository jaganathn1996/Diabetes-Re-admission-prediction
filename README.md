# Diabetes Re-admission Project

## Summary
This project aims to predict the likelihood of re-admission for diabetic patients using various machine learning techniques. The dataset used for this project is `diabetic_data.csv`, which contains 101,766 entries and 50 columns.

## Data Cleaning and Preprocessing
### Objective
The goal is to clean and preprocess the dataset to ensure it is suitable for machine learning models.

### Steps
1. **Identifying Missing Values**
   - Focused on `max_glu_serum` and `A1Cresult` columns.
   - Calculated the percentage of missing data for each column.

2. **Handling Missing Values**
   - `max_glu_serum`: 94.75% missing, recommended dropping.
   - `A1Cresult`: 83.28% missing, recommended dropping.
   - Imputed missing values in `admission_source`, `admission_type`, and `discharge_disposition` with "Not Available" or "Not Mapped".

3. **Handling Invalid Values**
   - Replaced invalid values (`?`) in columns like `race`, `payer_code`, and `medical_specialty` with the most frequent values.

4. **Dropping Rows and Columns**
   - Dropped rows with invalid `gender` values.
   - Dropped columns with high concentration in one category (>99%).

5. **Encoding Categorical Features**
   - Applied one-hot encoding, binary encoding, and label/ordinal encoding to various categorical columns.

6. **Data Grouping**
   - Grouped ICD-9 codes and medical specialties into broader categories.

## Model Training and Evaluation
### Datasets Considered
- **Cleaned Dataset**: All missing and invalid values handled manually.
- **Uncleaned Dataset**: Minimal preprocessing, handled by XGBoost.

### Models Used
- **XGBoost**: Applied to both cleaned and uncleaned datasets.

### Results
- **Accuracy**: Varied between 0.5915 and 0.6884 depending on the dataset and preprocessing applied.
- **Confusion Matrix, Precision, Recall, F1 Score**: Detailed metrics provided for each model configuration.

## Conclusion
The project demonstrates the importance of data cleaning and preprocessing in improving model performance. The cleaned dataset with cross-validation provided the best results.

## Files
- `diabetic_data.csv`: The dataset used for the project.
- `IDS_mapping.csv`: Mapping for `admission_type_id`, `discharge_disposition_id`, and `admission_source_id`.
- `Readmission_Summary_Jagan.docx`: Detailed summary of the project.

## How to Run
1. Clone the repository.
2. Install the required libraries: `pandas`, `numpy`, `xgboost`, etc.
3. Run the preprocessing script to clean the data.
4. Train the model using the cleaned dataset.
5. Evaluate the model performance.

