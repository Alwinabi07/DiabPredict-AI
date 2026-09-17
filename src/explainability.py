"""
Explainability utilities using SHAP for diabetes risk prediction.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')


class ModelExplainer:
    """
    SHAP-based model explainer for diabetes risk prediction.
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize explainer.
        
        Args:
            model: Trained model (must have predict_proba method)
            feature_names: List of feature names after preprocessing
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def fit(self, X_background: np.ndarray, max_samples: int = 100) -> None:
        """
        Fit SHAP explainer using background dataset.
        
        Args:
            X_background: Background data for SHAP (training data sample)
            max_samples: Maximum samples to use for background
        """
        # Sample background data if too large
        if len(X_background) > max_samples:
            idx = np.random.choice(len(X_background), max_samples, replace=False)
            X_background = X_background[idx]
        
        # Use TreeExplainer for tree-based models, otherwise KernelExplainer
        if hasattr(self.model, 'get_booster') or hasattr(self.model, 'estimators_'):
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                X_background,
                link="logit"
            )
    
    def explain_instance(self, X_instance: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a single instance.
        
        Args:
            X_instance: Single instance (1 x n_features)
            
        Returns:
            SHAP values for the instance
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        shap_values = self.explainer.shap_values(X_instance)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            # For binary classification, return positive class
            return shap_values[1] if len(shap_values) > 1 else shap_values[0]
        return shap_values
    
    def explain_dataset(self, X: np.ndarray, max_samples: int = 500) -> np.ndarray:
        """
        Compute SHAP values for a dataset.
        
        Args:
            X: Dataset to explain
            max_samples: Maximum samples to explain
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
            
        # Sample if too large
        if len(X) > max_samples:
            idx = np.random.choice(len(X), max_samples, replace=False)
            X = X[idx]
        
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            self.shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            self.shap_values = shap_values
            
        return self.shap_values
    
    def plot_shap_summary(self, X: np.ndarray, save_path: Optional[str] = None) -> None:
        """
        Plot SHAP summary (beeswarm) plot.
        
        Args:
            X: Data to explain
            save_path: Path to save plot
        """
        if self.shap_values is None:
            self.explain_dataset(X)
        
        shap.summary_plot(
            self.shap_values, 
            X, 
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_shap_bar(self, save_path: Optional[str] = None) -> None:
        """
        Plot mean absolute SHAP values (bar plot).
        
        Args:
            save_path: Path to save plot
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values computed. Call explain_dataset() first.")
        
        shap.summary_plot(
            self.shap_values, 
            feature_names=self.feature_names,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_waterfall(self, X_instance: np.ndarray, instance_idx: int = 0, save_path: Optional[str] = None) -> None:
        """
        Plot SHAP waterfall for a single instance.
        
        Args:
            X_instance: Single instance
            instance_idx: Index for labeling
            save_path: Path to save plot
        """
        shap_vals = self.explain_instance(X_instance)
        
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_vals[0],
                base_values=self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, np.ndarray) else self.explainer.expected_value,
                data=X_instance[0],
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on mean absolute SHAP values.
        
        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values computed. Call explain_dataset() first.")
        
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        return importance_df


def create_explainer(model, feature_names: List[str], X_background: np.ndarray) -> ModelExplainer:
    """
    Factory function to create and fit explainer.
    
    Args:
        model: Trained model
        feature_names: Feature names
        X_background: Background data
        
    Returns:
        Fitted ModelExplainer
    """
    explainer = ModelExplainer(model, feature_names)
    explainer.fit(X_background)
    return explainer