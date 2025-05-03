import pandas as pd
from typing import List, Dict

class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()  # Explicitly create a copy of the DataFrame

    def drop_columns(self, columns: List[str]):
        """
        Drops specified columns from the DataFrame.
        
        Args:
            columns (List[str]): List of column names to drop.
        """
        self.df.drop(columns, axis=1, inplace=True)

    def handle_null_values(self):
        """
        Handles null values in specific columns by filling them with predefined values.
        """
        self.df.loc[:, 'admission_source'] = self.df['admission_source'].fillna('Not Available')
        self.df.loc[:, 'admission_type'] = self.df['admission_type'].fillna('Not Available')
        self.df.loc[:, 'discharge_disposition'] = self.df['discharge_disposition'].fillna('Not Mapped')

    def replace_values(self):
        """
        Replaces specific values in certain columns.
        """
        self.df.loc[:, 'race'] = self.df['race'].replace('?', 'Caucasian')
        self.df.loc[:, 'payer_code'] = self.df['payer_code'].replace('?', 'MC')
        self.df.loc[:, 'medical_specialty'] = self.df['medical_specialty'].replace('?', 'InternalMedicine')

    def drop_invalid_rows(self):
        """
        Drops rows with invalid values in specific columns.
        """
        drop_condition = ((self.df['gender'] == 'Unknown/Invalid') | 
                          (self.df['diag_1'] == '?') | 
                          (self.df['diag_2'] == '?') | 
                          (self.df['diag_3'] == '?'))
        self.df = self.df.loc[~drop_condition].copy()  # Ensure a copy is made

    def ordinal_encode_age(self):
        """
        Ordinal encodes the 'age' column.
        """
        age_mapping = {'[0-10)': 1, '[10-20)': 2, '[20-30)': 3, '[30-40)': 4, 
                       '[40-50)': 5, '[50-60)': 6, '[60-70)': 7, '[70-80)': 8, 
                       '[80-90)': 9, '[90-100)': 10}
        self.df.loc[:, 'age'] = self.df['age'].map(age_mapping)

    def map_icd9_chapters(self, icd9_chapter_mapping: Dict[str, str]):
        """
        Maps ICD-9 codes to their respective chapters.
        
        Args:
            icd9_chapter_mapping (Dict[str, str]): Mapping of ICD-9 code ranges to chapter names.
        """
        def get_chapter(icd_code):
            if icd_code == "?":
                return "?"
            elif icd_code.startswith('E'):
                return icd9_chapter_mapping['E000-E999']
            elif icd_code.startswith('V'):
                return icd9_chapter_mapping['V01-V91']
            else:
                for range_key, chapter_name in icd9_chapter_mapping.items():
                    range_start, range_end = map(int, range_key.split('-'))
                    code_num = int(icd_code.split('.')[0])
                    if range_start <= code_num <= range_end:
                        return chapter_name
            return 'Other'

        self.df.loc[:, 'diag_1'] = self.df['diag_1'].apply(get_chapter)
        self.df.loc[:, 'diag_2'] = self.df['diag_2'].apply(get_chapter)
        self.df.loc[:, 'diag_3'] = self.df['diag_3'].apply(get_chapter)

    def categorize_specialty(self, grouping: Dict[str, List[str]]):
        """
        Categorizes the 'medical_specialty' column based on predefined groups.
        
        Args:
            grouping (Dict[str, List[str]]): Mapping of specialty groups to their respective specialties.
        """
        def categorize(specialty):
            for group, specialties in grouping.items():
                if specialty in specialties:
                    return group
            return specialty

        self.df.loc[:, 'medical_specialty'] = self.df['medical_specialty'].apply(categorize)

    def categorize_admission_source(self, admission_sources: Dict[str, List[str]]):
        """
        Categorizes the 'admission_source' column based on predefined groups.
        
        Args:
            admission_sources (Dict[str, List[str]]): Mapping of admission source groups to their respective sources.
        """
        def categorize(source):
            for group, sources in admission_sources.items():
                if source in sources:
                    return group
            return source

        self.df.loc[:, 'admission_source'] = self.df['admission_source'].str.strip()
        self.df.loc[:, 'admission_source'] = self.df['admission_source'].apply(categorize)

    def categorize_discharge_disposition(self, discharge_categories: Dict[str, List[str]]):
        """
        Categorizes the 'discharge_disposition' column based on predefined groups.
        
        Args:
            discharge_categories (Dict[str, List[str]]): Mapping of discharge disposition groups to their respective dispositions.
        """
        def categorize(disposition):
            for group, dispositions in discharge_categories.items():
                if disposition in dispositions:
                    return group
            return disposition

        self.df.loc[:, 'discharge_disposition'] = self.df['discharge_disposition'].str.strip()
        self.df.loc[:, 'discharge_disposition'] = self.df['discharge_disposition'].apply(categorize)

    def drop_high_concentration_columns(self, columns: List[str]):
        """
        Drops columns with high concentration of a single value.
        
        Args:
            columns (List[str]): List of column names to drop.
        """
        self.df.drop(columns, axis=1, inplace=True)

    def preprocess(self, icd9_chapter_mapping: Dict[str, str], grouping: Dict[str, List[str]], 
                   admission_sources: Dict[str, List[str]], discharge_categories: Dict[str, List[str]], 
                   columns_with_high_concentration: List[str]) -> pd.DataFrame:
        """
        Executes all preprocessing steps on the DataFrame.
        
        Args:
            icd9_chapter_mapping (Dict[str, str]): Mapping of ICD-9 code ranges to chapter names.
            grouping (Dict[str, List[str]]): Mapping of specialty groups to their respective specialties.
            admission_sources (Dict[str, List[str]]): Mapping of admission source groups to their respective sources.
            discharge_categories (Dict[str, List[str]]): Mapping of discharge disposition groups to their respective dispositions.
            columns_with_high_concentration (List[str]): List of column names to drop due to high concentration of a single value.
        
        Returns:
            pd.DataFrame: The preprocessed DataFrame.
        """
        self.drop_columns(['max_glu_serum', 'A1Cresult', 'weight'])
        self.handle_null_values()
        self.replace_values()
        self.drop_invalid_rows()
        self.ordinal_encode_age()
        self.map_icd9_chapters(icd9_chapter_mapping)
        self.categorize_specialty(grouping)
        self.categorize_admission_source(admission_sources)
        self.categorize_discharge_disposition(discharge_categories)
        self.drop_high_concentration_columns(columns_with_high_concentration)
        return self.df.copy()