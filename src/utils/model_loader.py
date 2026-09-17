"""
Model persistence and loading utilities for DiabPredict AI.

Provides cached loaders for Streamlit and other inference contexts.
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from functools import lru_cache

from ..predict import DiabetesRiskPredictor, BRFSS_FEATURES
from ..explainability import ModelExplainer, create_explainer
from ..preprocessing import BRFSSPreprocessor


# Default model directory
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')


@lru_cache(maxsize=1)
def load_best_model(model_dir: str = DEFAULT_MODEL_DIR):
    """
    Load the best trained model (Random Forest).
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        Trained model object
    """
    model_path = os.path.join(model_dir, "best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Best model not found at {model_path}")
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def load_preprocessor(model_dir: str = DEFAULT_MODEL_DIR) -> BRFSSPreprocessor:
    """
    Load the fitted preprocessor.
    
    Args:
        model_dir: Directory containing preprocessor file
        
    Returns:
        Fitted BRFSSPreprocessor
    """
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
    return joblib.load(preprocessor_path)


@lru_cache(maxsize=1)
def load_predictor(model_dir: str = DEFAULT_MODEL_DIR) -> DiabetesRiskPredictor:
    """
    Load the complete predictor (model + preprocessor).
    
    Args:
        model_dir: Directory containing model and preprocessor files
        
    Returns:
        DiabetesRiskPredictor instance ready for inference
    """
    return DiabetesRiskPredictor(
        model_path=os.path.join(model_dir, "best_model.joblib"),
        preprocessor_path=os.path.join(model_dir, "preprocessor.joblib")
    )


def load_all_models(model_dir: str = DEFAULT_MODEL_DIR) -> Dict[str, Any]:
    """
    Load all trained models for comparison.
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        Dictionary mapping model names to model objects
    """
    model_files = {
        'random_forest': 'random_forest.joblib',
        'xgboost': 'xgboost.joblib',
        'logistic_regression': 'logistic_regression.joblib',
        'decision_tree': 'decision_tree.joblib',
        'best_model': 'best_model.joblib'
    }
    
    models = {}
    for name, filename in model_files.items():
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


def get_training_sample(data_dir: str = DEFAULT_DATA_DIR, n_samples: int = 500, random_state: int = 42) -> np.ndarray:
    """
    Get a sample of training data for SHAP background.
    
    Args:
        data_dir: Directory containing raw data
        n_samples: Number of samples to return
        random_state: Random seed
        
    Returns:
        Transformed feature array for background
    """
    preprocessor = load_preprocessor()
    
    # Load raw data
    raw_path = os.path.join(data_dir, 'raw', 'diabetes_binary_health_indicators_BRFSS2015.csv')
    df = pd.read_csv(raw_path)
    df = df.drop_duplicates()
    
    # Stratified sample
    from sklearn.model_selection import train_test_split
    _, sample_df = train_test_split(
        df, test_size=n_samples/len(df), random_state=random_state,
        stratify=df['Diabetes_binary']
    )
    
    # Transform
    X_sample = preprocessor.transform(sample_df)
    return X_sample


@lru_cache(maxsize=1)
def load_shap_explainer(
    model_name: str = 'best_model',
    model_dir: str = DEFAULT_MODEL_DIR,
    data_dir: str = DEFAULT_DATA_DIR,
    max_background: int = 200
) -> ModelExplainer:
    """
    Load or create a fitted SHAP explainer.
    
    Args:
        model_name: Which model to explain ('best_model', 'random_forest', etc.)
        model_dir: Directory containing model files
        data_dir: Directory containing raw data for background
        max_background: Max samples for background dataset
        
    Returns:
        Fitted ModelExplainer
    """
    models = load_all_models(model_dir)
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available: {list(models.keys())}")
    
    model = models[model_name]
    preprocessor = load_preprocessor(model_dir)
    feature_names = preprocessor.get_feature_names_out()
    
    # Get background data
    X_background = get_training_sample(data_dir, n_samples=max_background)
    
    # Create and fit explainer
    explainer = create_explainer(model, feature_names, X_background)
    return explainer


def save_shap_explainer(
    explainer: ModelExplainer,
    model_name: str = 'best_model',
    model_dir: str = DEFAULT_MODEL_DIR
) -> str:
    """
    Save a fitted SHAP explainer to disk.
    
    Args:
        explainer: Fitted ModelExplainer
        model_name: Name for the saved file
        model_dir: Directory to save to
        
    Returns:
        Path to saved explainer
    """
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{model_name}_shap_explainer.joblib")
    joblib.dump(explainer, path)
    return path


@lru_cache(maxsize=1)
def load_or_create_shap_explainer(
    model_name: str = 'best_model',
    model_dir: str = DEFAULT_MODEL_DIR,
    data_dir: str = DEFAULT_DATA_DIR,
    max_background: int = 200
) -> ModelExplainer:
    """
    Load existing SHAP explainer or create new one if not found.
    
    Args:
        model_name: Which model to explain
        model_dir: Directory containing model files
        data_dir: Directory containing raw data
        max_background: Max samples for background
        
    Returns:
        Fitted ModelExplainer
    """
    explainer_path = os.path.join(model_dir, f"{model_name}_shap_explainer.joblib")
    
    if os.path.exists(explainer_path):
        try:
            return joblib.load(explainer_path)
        except Exception:
            pass  # Fall through to create new
    
    explainer = load_shap_explainer(model_name, model_dir, data_dir, max_background)
    save_shap_explainer(explainer, model_name, model_dir)
    return explainer


def load_population_data(data_dir: str = DEFAULT_DATA_DIR) -> Dict[str, pd.DataFrame]:
    """
    Load all population analysis CSV reports.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        Dictionary mapping report names to DataFrames
    """
    reports_dir = os.path.join(os.path.dirname(data_dir), 'outputs', 'reports')
    
    reports = {}
    if os.path.exists(reports_dir):
        for file in os.listdir(reports_dir):
            if file.endswith('.csv'):
                name = file.replace('.csv', '').replace('prevalence_by_', '').replace('_', ' ')
                path = os.path.join(reports_dir, file)
                reports[name] = pd.read_csv(path)
    return reports


def load_forecast_data(data_dir: str = DEFAULT_DATA_DIR) -> Dict[str, pd.DataFrame]:
    """
    Load forecasting data and results.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        Dictionary with forecast data
    """
    processed_dir = os.path.join(data_dir, 'processed')
    reports_dir = os.path.join(os.path.dirname(data_dir), 'outputs', 'reports')
    
    data = {}
    
    # Main forecasting data
    main_path = os.path.join(processed_dir, 'india_forecasting_data.csv')
    if os.path.exists(main_path):
        data['india_series'] = pd.read_csv(main_path)
    
    # Future forecast
    future_path = os.path.join(reports_dir, 'arima_future_forecast.csv')
    if os.path.exists(future_path):
        data['future_forecast'] = pd.read_csv(future_path)
    
    # Test predictions
    test_path = os.path.join(reports_dir, 'arima_test_predictions.csv')
    if os.path.exists(test_path):
        data['test_predictions'] = pd.read_csv(test_path)
    
    # Validation predictions for all models
    for model in ['naive', 'linear_trend', 'arima', 'xgboost']:
        val_path = os.path.join(reports_dir, f'{model}_validation_predictions.csv')
        if os.path.exists(val_path):
            data[f'{model}_validation'] = pd.read_csv(val_path)
    
    return data


def load_model_comparison(data_dir: str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """
    Load model comparison results.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        Comparison DataFrame
    """
    reports_dir = os.path.join(os.path.dirname(data_dir), 'outputs', 'reports')
    
    # Try validation comparison first
    val_path = os.path.join(reports_dir, 'model_comparison_validation.csv')
    if os.path.exists(val_path):
        return pd.read_csv(val_path)
    
    # Fallback to training comparison
    train_path = os.path.join(reports_dir, 'model_comparison_report.csv')
    if os.path.exists(train_path):
        return pd.read_csv(train_path)
    
    return pd.DataFrame()


def verify_model_artifacts(model_dir: str = DEFAULT_MODEL_DIR) -> Dict[str, bool]:
    """
    Verify all required model artifacts exist.
    
    Args:
        model_dir: Directory to check
        
    Returns:
        Dictionary of artifact names and existence status
    """
    artifacts = {
        'best_model': 'best_model.joblib',
        'preprocessor': 'preprocessor.joblib',
        'random_forest': 'random_forest.joblib',
        'xgboost': 'xgboost.joblib',
        'logistic_regression': 'logistic_regression.joblib',
        'decision_tree': 'decision_tree.joblib'
    }
    
    status = {}
    for name, filename in artifacts.items():
        path = os.path.join(model_dir, filename)
        status[name] = os.path.exists(path)
    
    return status


def test_inference(predictor: DiabetesRiskPredictor) -> Dict[str, Any]:
    """
    Run a test inference to verify the predictor works.
    
    Args:
        predictor: DiabetesRiskPredictor instance
        
    Returns:
        Dictionary with test results
    """
    # Sample input matching BRFSS features
    test_input = {
        'HighBP': 1, 'HighChol': 1, 'CholCheck': 1, 'Smoker': 0,
        'Stroke': 0, 'HeartDiseaseorAttack': 0, 'PhysActivity': 1,
        'Fruits': 1, 'Veggies': 1, 'HvyAlcoholConsump': 0,
        'AnyHealthcare': 1, 'NoDocbcCost': 0, 'DiffWalk': 0,
        'Sex': 1, 'GenHlth': 3, 'Age': 7, 'Education': 4,
        'Income': 5, 'BMI': 28.0, 'MentHlth': 2, 'PhysHlth': 1
    }
    
    try:
        proba = predictor.predict_proba(test_input)
        pred_class = predictor.predict(test_input)
        category = predictor.get_risk_category(proba)
        
        return {
            'success': True,
            'probability': proba,
            'predicted_class': pred_class,
            'risk_category': category,
            'input': test_input
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'input': test_input
        }


if __name__ == "__main__":
    # Test model loading and inference
    print("Testing model loading...")
    
    status = verify_model_artifacts()
    print("Model artifacts:")
    for name, exists in status.items():
        print(f"  {name}: {'OK' if exists else 'MISSING'}")
    
    print("\nLoading predictor...")
    predictor = load_predictor()
    print("Predictor loaded successfully")
    
    print("\nRunning test inference...")
    result = test_inference(predictor)
    if result['success']:
        print(f"  Probability: {result['probability']:.4f}")
        print(f"  Class: {result['predicted_class']}")
        print(f"  Risk Category: {result['risk_category']}")
    else:
        print(f"  ERROR: {result['error']}")