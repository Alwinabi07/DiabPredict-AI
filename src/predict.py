"""
Prediction utilities for individual diabetes risk prediction.

Uses the trained CDC BRFSS model with 21 features:
- Binary (14): HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack,
  PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost,
  DiffWalk, Sex
- Ordinal (4): GenHlth, Age, Education, Income
- Continuous (3): BMI, MentHlth, PhysHlth
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional, List
from .preprocessing import BRFSSPreprocessor


# The 21 features expected by the trained model (in order)
BRFSS_FEATURES = [
    'HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke',
    'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
    'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex',
    'GenHlth', 'Age', 'Education', 'Income',
    'BMI', 'MentHlth', 'PhysHlth'
]

# Human-readable labels for UI
FEATURE_LABELS = {
    'HighBP': 'High Blood Pressure',
    'HighChol': 'High Cholesterol',
    'CholCheck': 'Cholesterol Check in 5 Years',
    'Smoker': 'Smoker',
    'Stroke': 'Stroke',
    'HeartDiseaseorAttack': 'Heart Disease or Heart Attack',
    'PhysActivity': 'Physical Activity',
    'Fruits': 'Fruit Consumption (1+ per day)',
    'Veggies': 'Vegetable Consumption (1+ per day)',
    'HvyAlcoholConsump': 'Heavy Alcohol Consumption',
    'AnyHealthcare': 'Healthcare Coverage',
    'NoDocbcCost': 'Could Not Afford Doctor',
    'DiffWalk': 'Difficulty Walking',
    'Sex': 'Sex',
    'GenHlth': 'General Health',
    'Age': 'Age Category',
    'Education': 'Education Level',
    'Income': 'Income Level',
    'BMI': 'Body Mass Index (BMI)',
    'MentHlth': 'Mental Health Days (past 30)',
    'PhysHlth': 'Physical Health Days (past 30)'
}

# Value options for categorical/ordinal features
FEATURE_OPTIONS = {
    'HighBP': {0: 'No', 1: 'Yes'},
    'HighChol': {0: 'No', 1: 'Yes'},
    'CholCheck': {0: 'No', 1: 'Yes'},
    'Smoker': {0: 'No', 1: 'Yes'},
    'Stroke': {0: 'No', 1: 'Yes'},
    'HeartDiseaseorAttack': {0: 'No', 1: 'Yes'},
    'PhysActivity': {0: 'No', 1: 'Yes'},
    'Fruits': {0: 'No', 1: 'Yes'},
    'Veggies': {0: 'No', 1: 'Yes'},
    'HvyAlcoholConsump': {0: 'No', 1: 'Yes'},
    'AnyHealthcare': {0: 'No', 1: 'Yes'},
    'NoDocbcCost': {0: 'No', 1: 'Yes'},
    'DiffWalk': {0: 'No', 1: 'Yes'},
    'Sex': {0: 'Female', 1: 'Male'},
    'GenHlth': {1: 'Excellent', 2: 'Very Good', 3: 'Good', 4: 'Fair', 5: 'Poor'},
    'Age': {
        1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39', 5: '40-44',
        6: '45-49', 7: '50-54', 8: '55-59', 9: '60-64', 10: '65-69',
        11: '70-74', 12: '75-79', 13: '80+'
    },
    'Education': {
        1: 'Never attended / Kindergarten',
        2: 'Elementary',
        3: 'Some High School',
        4: 'High School Graduate',
        5: 'Some College / Technical School',
        6: 'College Graduate'
    },
    'Income': {
        1: 'Less than $10,000',
        2: '$10,000 - $15,000',
        3: '$15,000 - $20,000',
        4: '$20,000 - $25,000',
        5: '$25,000 - $35,000',
        6: '$35,000 - $50,000',
        7: '$50,000 - $75,000',
        8: '$75,000 or more'
    }
}

# Numeric ranges for continuous features
FEATURE_RANGES = {
    'BMI': {'min': 12.0, 'max': 98.0, 'mean': 28.0},
    'MentHlth': {'min': 0, 'max': 30, 'mean': 3},
    'PhysHlth': {'min': 0, 'max': 30, 'mean': 3}
}


class DiabetesRiskPredictor:
    """
    Wrapper for trained diabetes risk prediction model.
    """
    
    def __init__(self, model_path: str, preprocessor_path: str):
        """
        Load model and preprocessor.
        
        Args:
            model_path: Path to saved model
            preprocessor_path: Path to saved preprocessor
        """
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.feature_names = self.preprocessor.get_feature_names_out()
    
    def _validate_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Validate and convert input data to DataFrame with correct features.
        
        Args:
            input_data: Dictionary of feature values
            
        Returns:
            DataFrame with all 21 features in correct order
        """
        # Check all required features present
        missing = [f for f in BRFSS_FEATURES if f not in input_data]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Create DataFrame with features in correct order
        df = pd.DataFrame([[input_data[f] for f in BRFSS_FEATURES]], columns=BRFSS_FEATURES)
        return df
    
    def predict_proba(self, input_data: Dict[str, Any]) -> float:
        """
        Predict diabetes risk probability for a single individual.
        
        Args:
            input_data: Dictionary of feature values (21 BRFSS features)
            
        Returns:
            Probability of diabetes risk (0-1)
        """
        df = self._validate_input(input_data)
        X_transformed = self.preprocessor.transform(df)
        proba = self.model.predict_proba(X_transformed)[0, 1]
        return float(proba)
    
    def predict(self, input_data: Dict[str, Any], threshold: float = 0.5) -> int:
        """
        Predict diabetes risk class.
        
        Args:
            input_data: Dictionary of feature values
            threshold: Decision threshold
            
        Returns:
            Predicted class (0 or 1)
        """
        proba = self.predict_proba(input_data)
        return int(proba >= threshold)
    
    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities for a batch of individuals.
        
        Args:
            df: DataFrame with feature columns (must contain all 21 BRFSS features)
            
        Returns:
            Array of probabilities
        """
        # Ensure columns in correct order
        df = df[BRFSS_FEATURES]
        X_transformed = self.preprocessor.transform(df)
        return self.model.predict_proba(X_transformed)[:, 1]
    
    def get_risk_category(self, probability: float) -> str:
        """
        Convert probability to risk category.
        
        Args:
            probability: Predicted probability
            
        Returns:
            Risk category string
        """
        if probability < 0.2:
            return "Low Risk"
        elif probability < 0.5:
            return "Moderate Risk"
        elif probability < 0.8:
            return "High Risk"
        else:
            return "Very High Risk"


def load_predictor(model_dir: str = "models") -> DiabetesRiskPredictor:
    """
    Load predictor from model directory.
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        DiabetesRiskPredictor instance
    """
    import os
    model_path = os.path.join(model_dir, "best_model.joblib")
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    return DiabetesRiskPredictor(model_path, preprocessor_path)


def what_if_simulation(
    predictor: DiabetesRiskPredictor,
    base_input: Dict[str, Any],
    feature_to_vary: str,
    values: List[Any]
) -> pd.DataFrame:
    """
    Perform what-if simulation by varying one feature.
    
    Args:
        predictor: DiabetesRiskPredictor instance
        base_input: Base input dictionary
        feature_to_vary: Feature name to vary
        values: List of values to test
        
    Returns:
        DataFrame with simulation results
    """
    results = []
    
    for value in values:
        test_input = base_input.copy()
        test_input[feature_to_vary] = value
        prob = predictor.predict_proba(test_input)
        category = predictor.get_risk_category(prob)
        results.append({
            feature_to_vary: value,
            'probability': prob,
            'risk_category': category
        })
    
    return pd.DataFrame(results)


def get_brfss_feature_info() -> Dict[str, Dict]:
    """
    Get complete feature information for UI building.
    
    Returns:
        Dictionary with feature metadata for all 21 features
    """
    info = {}
    for feat in BRFSS_FEATURES:
        info[feat] = {
            'label': FEATURE_LABELS.get(feat, feat),
            'type': 'binary' if feat in FEATURE_OPTIONS and len(FEATURE_OPTIONS[feat]) == 2 else 
                   'ordinal' if feat in FEATURE_OPTIONS else 'continuous',
            'options': FEATURE_OPTIONS.get(feat, None),
            'range': FEATURE_RANGES.get(feat, None)
        }
    return info