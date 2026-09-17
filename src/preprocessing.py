"""
Data preprocessing utilities for DiabPredict AI.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from typing import List, Tuple, Optional, Dict
from sklearn.model_selection import train_test_split


class BRFSSPreprocessor:
    """
    Preprocessor for BRFSS diabetes dataset.
    
    Handles feature types appropriately:
    - Binary (0/1): HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack,
      PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost,
      DiffWalk, Sex
    - Ordinal: GenHlth (1-5), Age (1-13), Education (1-6), Income (1-8)
    - Continuous: BMI (12-98), MentHlth (0-30), PhysHlth (0-30)
    """
    
    # Feature type definitions for BRFSS dataset
    BINARY_FEATURES = [
        'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex'
    ]
    
    ORDINAL_FEATURES = [
        'GenHlth', 'Age', 'Education', 'Income'
    ]
    
    CONTINUOUS_FEATURES = [
        'BMI', 'MentHlth', 'PhysHlth'
    ]
    
    ALL_FEATURES = BINARY_FEATURES + ORDINAL_FEATURES + CONTINUOUS_FEATURES
    
    def __init__(self, random_state: int = 42, scale_for_linear: bool = True):
        self.random_state = random_state
        self.scale_for_linear = scale_for_linear
        self.preprocessor = None
        self.feature_names_out = None
        self.is_fitted = False
        
    def build_preprocessor(self) -> ColumnTransformer:
        """
        Build sklearn ColumnTransformer for preprocessing.
        
        For linear models: scale continuous features, pass binary/ordinal through
        For tree models: scaling not needed but harmless
        """
        transformers = []
        
        # Binary features: impute only (no scaling needed)
        transformers.append((
            'binary',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent'))
            ]),
            self.BINARY_FEATURES
        ))
        
        # Ordinal features: impute only (preserve order)
        transformers.append((
            'ordinal',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median'))
            ]),
            self.ORDINAL_FEATURES
        ))
        
        # Continuous features: impute and optionally scale
        if self.scale_for_linear:
            cont_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
        else:
            cont_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median'))
            ])
        transformers.append(('continuous', cont_pipeline, self.CONTINUOUS_FEATURES))
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',  # Drop any columns not specified
            verbose_feature_names_out=False
        )
        
        return preprocessor
    
    def fit(self, df: pd.DataFrame) -> 'BRFSSPreprocessor':
        """
        Fit preprocessor on training data.
        
        Args:
            df: Training dataframe with features and target
            
        Returns:
            self
        """
        X = df[self.ALL_FEATURES]
        self.preprocessor = self.build_preprocessor()
        self.preprocessor.fit(X)
        self.feature_names_out = self.preprocessor.get_feature_names_out().tolist()
        self.is_fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            df: Dataframe with features
            
        Returns:
            Transformed feature array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        X = df[self.ALL_FEATURES]
        return self.preprocessor.transform(X)
    
    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def get_feature_names_out(self) -> List[str]:
        """Get feature names after transformation."""
        if not self.is_fitted:
            return self.ALL_FEATURES
        return self.feature_names_out


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load raw data and remove duplicates.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Cleaned dataframe
    """
    df = pd.read_csv(filepath)
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    print(f"Removed {removed} duplicate rows ({removed/initial_rows*100:.1f}%)")
    return df


def stratified_train_test_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split preserving class distribution.
    
    Args:
        df: Input dataframe
        target_col: Target column name
        test_size: Fraction for test set
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col]
    )
    
    print(f"Train set: {len(train_df)} rows, Target dist: {train_df[target_col].value_counts(normalize=True).to_dict()}")
    print(f"Test set: {len(test_df)} rows, Target dist: {test_df[target_col].value_counts(normalize=True).to_dict()}")
    
    return train_df, test_df


def prepare_data(
    filepath: str,
    target_col: str = 'Diabetes_binary',
    test_size: float = 0.2,
    random_state: int = 42,
    scale_for_linear: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, BRFSSPreprocessor, pd.DataFrame, pd.DataFrame]:
    """
    Complete data preparation pipeline.
    
    Args:
        filepath: Path to raw CSV
        target_col: Target column name
        test_size: Test set fraction
        random_state: Random seed
        scale_for_linear: Whether to scale continuous features
        
    Returns:
        X_train, X_test, y_train, y_test, preprocessor, train_df, test_df
    """
    # Load and clean
    df = load_and_clean_data(filepath)
    
    # Stratified split
    train_df, test_df = stratified_train_test_split(
        df, target_col, test_size, random_state
    )
    
    # Fit preprocessor on training data only
    preprocessor = BRFSSPreprocessor(random_state=random_state, scale_for_linear=scale_for_linear)
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Feature names: {preprocessor.get_feature_names_out()}")
    
    return X_train, X_test, y_train, y_test, preprocessor, train_df, test_df


if __name__ == "__main__":
    # Test preprocessing
    X_train, X_test, y_train, y_test, preprocessor, train_df, test_df = prepare_data(
        r'C:\Users\ALWIN ABHISHEK\DiabPredict-AI\data\raw\diabetes_binary_health_indicators_BRFSS2015.csv'
    )