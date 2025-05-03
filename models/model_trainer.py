import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
from typing import Tuple
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

class ModelTrainer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def prepare_data(self, one_hot_encode: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepares the data for training and testing.

        Args:
            one_hot_encode (bool): Whether to apply one-hot encoding to categorical features.

        Returns:
            Tuple containing training and testing data and labels.
        """
        X_cleaned = self.df.drop('readmitted', axis=1).copy()
        y_cleaned = self.df['readmitted'].copy()
        X_cleaned.columns = X_cleaned.columns.str.replace('-', '_')
        
        if one_hot_encode:
            object_columns = X_cleaned.select_dtypes('object').columns
            X_cleaned = pd.get_dummies(X_cleaned, columns=object_columns)
        
        y_cleaned = y_cleaned.map({'<30': 0, '>30': 0, 'NO': 1})
        X_cleaned_train, X_cleaned_test, y_cleaned_train, y_cleaned_test = train_test_split(
            X_cleaned, y_cleaned, random_state=42, stratify=y_cleaned
        )
        return X_cleaned_train, X_cleaned_test, y_cleaned_train, y_cleaned_test

    def prepare_data_with_missing_values(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepares the data for training and testing without manually handling missing values.

        Returns:
            Tuple containing training and testing data and labels.
        """
        df_valmissing = self.df.copy()
        df_valmissing = df_valmissing.replace(["?", 'Unknown/Invalid'], np.nan)
        
        # Ordinal encoding for age column
        df_valmissing['age'] = df_valmissing['age'].map({'[0-10)': 1, '[10-20)': 2, '[20-30)': 3, '[30-40)': 4, 
                                                         '[40-50)': 5, '[50-60)': 6, '[60-70)': 7, '[70-80)': 8, 
                                                         '[80-90)': 9, '[90-100)': 10})
        
        # Dropping specific columns
        df_valmissing.drop(['max_glu_serum', 'A1Cresult', 'weight'], axis=1, inplace=True)
        df_valmissing.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True)
        
        df_valmissing.replace(' ', '_', regex=True, inplace=True)
        
        X = df_valmissing.drop('readmitted', axis=1).copy()
        y = df_valmissing['readmitted'].copy()
        X.columns = X.columns.str.replace('-', '_')
        
        object_columns = X.select_dtypes('object').columns
        X_encoded = pd.get_dummies(X, columns=object_columns)
        y = y.map({'<30': 0, '>30': 0, 'NO': 1})
        
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test

    def prepare_data_without_one_hot_encoding(self, cleaned: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepares the data for training and testing without one-hot encoding.

        Args:
            cleaned (bool): Whether the dataset is cleaned or not.

        Returns:
            Tuple containing training and testing data and labels.
        """
        if cleaned:
            X = self.df.drop('readmitted', axis=1).copy()
            y = self.df['readmitted'].copy()
        else:
            df_valmissing = self.df.copy()
            df_valmissing = df_valmissing.replace(["?", 'Unknown/Invalid'], np.nan)
            
            # Ordinal encoding for age column
            df_valmissing['age'] = df_valmissing['age'].map({'[0-10)': 1, '[10-20)': 2, '[20-30)': 3, '[30-40)': 4, 
                                                             '[40-50)': 5, '[50-60)': 6, '[60-70)': 7, '[70-80)': 8, 
                                                             '[80-90)': 9, '[90-100)': 10})
            
            # Dropping specific columns
            df_valmissing.drop(['max_glu_serum', 'A1Cresult', 'weight'], axis=1, inplace=True)
            df_valmissing.drop(['encounter_id', 'patient_nbr'], axis=1, inplace=True)
            
            df_valmissing.replace(' ', '_', regex=True, inplace=True)
            
            X = df_valmissing.drop('readmitted', axis=1).copy()
            y = df_valmissing['readmitted'].copy()
        
        X.columns = X.columns.str.replace('-', '_')
        
        for col in X.select_dtypes('object').columns:
            X[col] = X[col].astype('category')
        
        y = y.map({'<30': 0, '>30': 0, 'NO': 1})
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame = None, y_test: pd.Series = None, enable_categorical: bool = False):
        """
        Trains the XGBoost model.

        Args:
            X_train (pd.DataFrame): Training data.
            y_train (pd.Series): Training labels.
            X_test (pd.DataFrame): Testing data.
            y_test (pd.Series): Testing labels.
            enable_categorical (bool): Whether to enable categorical data handling.

        Returns:
            Trained XGBoost model.
        """
        clf_xgb = xgb.XGBClassifier(objective='binary:logistic', early_stopping_rounds=10, eval_metric='aucpr', seed=42, enable_categorical=enable_categorical)
        
        clf_xgb.fit(X_train, y_train, verbose=False, eval_set=[(X_test, y_test)] if X_test is not None else None)
        
        if X_test is not None and y_test is not None:
            y_pred = clf_xgb.predict(X_test)
            self.eval_metrics(y_test, y_pred)
        
        return clf_xgb

    def predict(self, model, X_test: pd.DataFrame) -> pd.Series:
        """
        Makes predictions using the trained model.

        Args:
            model: Trained model.
            X_test (pd.DataFrame): Testing data.

        Returns:
            pd.Series: Predictions.
        """
        return model.predict(X_test)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5):
        """
        Performs cross-validation on the dataset.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series): Labels.
            cv (int): Number of cross-validation folds.

        Returns:
            Cross-validation scores.
        """
        clf_xgb = xgb.XGBClassifier(objective='binary:logistic', eval_metric='aucpr', seed=42)
        scores = cross_val_score(clf_xgb, X, y, cv=cv, scoring='accuracy')
        return scores

    def modelfit(self, alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
        """
        Fits the model using cross-validation.

        Args:
            alg: The model to fit.
            dtrain: The training data.
            predictors: The predictor columns.
            useTrainCV (bool): Whether to use cross-validation.
            cv_folds (int): Number of cross-validation folds.
            early_stopping_rounds (int): Early stopping rounds for cross-validation.
        """
        if useTrainCV:
            xgb_param = alg.get_xgb_params()
            xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain['readmitted'].values)
            cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                              metrics='auc', early_stopping_rounds=early_stopping_rounds)
            alg.set_params(n_estimators=cvresult.shape[0])

        # Fit the algorithm on data
        alg.fit(dtrain[predictors], dtrain['readmitted'])

        # Predict training set
        dtrain_predictions = alg.predict(dtrain[predictors])

        # Print model report
        self.eval_metrics(dtrain['readmitted'].values, dtrain_predictions)

    def eval_metrics(self, y_test, y_pred):
        """
        Evaluates the model performance.

        Args:
            y_test: True labels.
            y_pred: Predicted labels.
        """
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Print metrics
        print("\nModel Report")
        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

    def prepare_data_for_cross_validation(self, cleaned: bool = True) -> pd.DataFrame:
        """
        Prepares the data for cross-validation.

        Args:
            cleaned (bool): Whether the dataset is cleaned or not.

        Returns:
            pd.DataFrame: The prepared DataFrame.
        """
        if cleaned:
            df = self.df.copy()
            df['readmitted'] = df['readmitted'].map({'<30': 0, '>30': 0, 'NO': 1})
            dtrain = pd.get_dummies(df, drop_first=True)
        else:
            df_valmissing = self.df.copy()
            df_valmissing['readmitted'] = df_valmissing['readmitted'].map({'<30': 0, '>30': 0, 'NO': 1})
            dtrain = pd.get_dummies(df_valmissing, drop_first=True)
        
        # Clean feature names
        dtrain.columns = dtrain.columns.str.replace('[', '', regex=False).str.replace(']', '', regex=False).str.replace('<', '', regex=False).str.replace('>', '', regex=False)
        
        return dtrain

    def train_with_cross_validation(self, dtrain: pd.DataFrame, target: str = 'readmitted'):
        """
        Trains the model using cross-validation.

        Args:
            dtrain (pd.DataFrame): The training data.
            target (str): The target column name.
        """
        predictors = [x for x in dtrain.columns if x not in [target]]
        xgb_model = xgb.XGBClassifier(
            learning_rate=0.1,
            n_estimators=1000,
            max_depth=5,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='auc',
            nthread=4,
            scale_pos_weight=1,
            seed=27
        )
        self.modelfit(xgb_model, dtrain, predictors)