"""
Training script for individual diabetes risk prediction model.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

from .preprocessing import BRFSSPreprocessor, prepare_data


def calculate_scale_pos_weight(y: np.ndarray) -> float:
    """
    Calculate scale_pos_weight for XGBoost based on class imbalance.
    
    Args:
        y: Target array
        
    Returns:
        scale_pos_weight value (neg/pos ratio)
    """
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return neg / pos if pos > 0 else 1.0


def get_models(random_state: int = 42, scale_pos_weight: float = 1.0) -> Dict[str, Any]:
    """
    Get dictionary of models to compare with appropriate class imbalance handling.
    
    Class imbalance handling:
    - Logistic Regression: class_weight='balanced' (adjusts C per class)
    - Decision Tree: class_weight='balanced' (adjusts split criterion)
    - Random Forest: class_weight='balanced' (adjusts bootstrap sampling)
    - XGBoost: scale_pos_weight = neg/pos ratio (weights positive class)
    
    Args:
        random_state: Random seed for reproducibility
        scale_pos_weight: Weight for positive class in XGBoost
        
    Returns:
        Dictionary mapping model names to model instances
    """
    return {
        'logistic_regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        ),
        'decision_tree': DecisionTreeClassifier(
            class_weight='balanced',
            random_state=random_state,
            max_depth=10  # Prevent overfitting
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5
        ),
        'xgboost': XGBClassifier(
            n_estimators=300,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            eval_metric='logloss',
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8
        )
    }


def train_and_evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models: Dict[str, Any],
    cv_folds: int = 5
) -> Dict[str, Dict[str, Any]]:
    """
    Train multiple models and evaluate with cross-validation using multiple metrics.
    
    Args:
        X_train: Training features
        y_train: Training targets
        models: Dictionary of models to train
        cv_folds: Number of CV folds
        
    Returns:
        Dictionary of results for each model
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    scoring = {
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation on training set with multiple metrics
        cv_results = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring,
            return_train_score=False, n_jobs=-1
        )
        
        # Fit on full training set
        model.fit(X_train, y_train)
        
        results[name] = {
            'cv_roc_auc_mean': cv_results['test_roc_auc'].mean(),
            'cv_roc_auc_std': cv_results['test_roc_auc'].std(),
            'cv_pr_auc_mean': cv_results['test_pr_auc'].mean(),
            'cv_pr_auc_std': cv_results['test_pr_auc'].std(),
            'cv_precision_mean': cv_results['test_precision'].mean(),
            'cv_precision_std': cv_results['test_precision'].std(),
            'cv_recall_mean': cv_results['test_recall'].mean(),
            'cv_recall_std': cv_results['test_recall'].std(),
            'cv_f1_mean': cv_results['test_f1'].mean(),
            'cv_f1_std': cv_results['test_f1'].std(),
            'model': model
        }
        
        print(f"  CV ROC-AUC: {cv_results['test_roc_auc'].mean():.4f} (+/- {cv_results['test_roc_auc'].std():.4f})")
        print(f"  CV PR-AUC:  {cv_results['test_pr_auc'].mean():.4f} (+/- {cv_results['test_pr_auc'].std():.4f})")
        print(f"  CV Precision: {cv_results['test_precision'].mean():.4f} (+/- {cv_results['test_precision'].std():.4f})")
        print(f"  CV Recall:    {cv_results['test_recall'].mean():.4f} (+/- {cv_results['test_recall'].std():.4f})")
        print(f"  CV F1:        {cv_results['test_f1'].mean():.4f} (+/- {cv_results['test_f1'].std():.4f})")
    
    return results


def select_best_model(results: Dict[str, Dict[str, Any]], metric: str = 'cv_roc_auc_mean') -> Tuple[str, Any]:
    """
    Select best model based on specified metric.
    
    Args:
        results: Results dictionary from train_and_evaluate_models
        metric: Metric to optimize
        
    Returns:
        Tuple of (best_model_name, best_model)
    """
    best_name = max(results.keys(), key=lambda k: results[k][metric])
    best_model = results[best_name]['model']
    print(f"\nBest model (by {metric}): {best_name} ({metric}={results[best_name][metric]:.4f})")
    return best_name, best_model


def evaluate_on_test_set(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate model on held-out test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of test metrics
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, roc_auc_score,
        average_precision_score, confusion_matrix
    )
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'test_precision': precision_score(y_test, y_pred, zero_division=0),
        'test_recall': recall_score(y_test, y_pred, zero_division=0),
        'test_f1': f1_score(y_test, y_pred, zero_division=0),
        'test_roc_auc': roc_auc_score(y_test, y_pred_proba),
        'test_pr_auc': average_precision_score(y_test, y_pred_proba)
    }
    
    cm = confusion_matrix(y_test, y_pred)
    metrics['test_tn'] = int(cm[0, 0])
    metrics['test_fp'] = int(cm[0, 1])
    metrics['test_fn'] = int(cm[1, 0])
    metrics['test_tp'] = int(cm[1, 1])
    
    return metrics


def save_model(model: Any, filepath: str) -> None:
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def save_preprocessor(preprocessor: BRFSSPreprocessor, filepath: str) -> None:
    """Save preprocessor to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(preprocessor, filepath)
    print(f"Preprocessor saved to {filepath}")


def load_model(filepath: str) -> Any:
    """Load trained model from disk."""
    return joblib.load(filepath)


def load_preprocessor(filepath: str) -> BRFSSPreprocessor:
    """Load preprocessor from disk."""
    return joblib.load(filepath)


def run_training_pipeline(
    data_path: str,
    output_dir: str = 'models',
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Run complete training pipeline.
    
    Args:
        data_path: Path to raw CSV data
        output_dir: Directory to save models
        test_size: Test set fraction
        random_state: Random seed
        cv_folds: Number of CV folds
        
    Returns:
        Dictionary with results, best model name, and test metrics
    """
    print("=" * 60)
    print("DIABETES RISK PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Prepare data
    print("\n1. PREPARING DATA...")
    X_train, X_test, y_train, y_test, preprocessor, train_df, test_df = prepare_data(
        data_path,
        target_col='Diabetes_binary',
        test_size=test_size,
        random_state=random_state,
        scale_for_linear=True
    )
    
    # Calculate class imbalance weight for XGBoost
    scale_pos_weight = calculate_scale_pos_weight(y_train)
    print(f"\nClass imbalance ratio (neg/pos): {scale_pos_weight:.2f}")
    print(f"Training set - Positive class: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
    print(f"Test set - Positive class: {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")
    
    # Get models
    models = get_models(random_state=random_state, scale_pos_weight=scale_pos_weight)
    
    # Train and cross-validate
    print("\n2. CROSS-VALIDATION ON TRAINING SET...")
    results = train_and_evaluate_models(X_train, y_train, models, cv_folds)
    
    # Select best model by CV ROC-AUC
    print("\n3. SELECTING BEST MODEL...")
    best_name, best_model = select_best_model(results, 'cv_roc_auc_mean')
    
    # Evaluate all models on test set
    print("\n4. EVALUATING ON HELD-OUT TEST SET...")
    test_results = {}
    for name, res in results.items():
        test_metrics = evaluate_on_test_set(res['model'], X_test, y_test)
        res.update(test_metrics)
        test_results[name] = test_metrics
        print(f"\n{name} TEST METRICS:")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    
    # Save models and preprocessor
    print("\n5. SAVING MODELS AND PREPROCESSOR...")
    os.makedirs(output_dir, exist_ok=True)
    
    for name, res in results.items():
        model_path = os.path.join(output_dir, f"{name}.joblib")
        save_model(res['model'], model_path)
    
    preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
    save_preprocessor(preprocessor, preprocessor_path)
    
    # Save best model separately
    best_model_path = os.path.join(output_dir, "best_model.joblib")
    save_model(best_model, best_model_path)
    
    return {
        'results': results,
        'best_model_name': best_name,
        'best_model': best_model,
        'preprocessor': preprocessor,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_df': train_df,
        'test_df': test_df
    }


if __name__ == "__main__":
    data_path = r'C:\Users\ALWIN ABHISHEK\DiabPredict-AI\data\raw\diabetes_binary_health_indicators_BRFSS2015.csv'
    output_dir = r'C:\Users\ALWIN ABHISHEK\DiabPredict-AI\models'
    
    run_training_pipeline(data_path, output_dir)