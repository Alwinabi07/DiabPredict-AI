"""
Evaluation utilities for diabetes risk prediction models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from typing import Dict, Tuple, Optional, List
import warnings
import os
warnings.filterwarnings('ignore')


def evaluate_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for binary classifier.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities for positive class
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba)
    }
    return metrics


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print detailed classification report."""
    print(classification_report(y_true, y_pred, target_names=['No Diabetes', 'Diabetes'], zero_division=0))


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    save_path: Optional[str] = None
) -> None:
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    title = 'Confusion Matrix'
    if model_name:
        title += f' - {model_name}'
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(
    results: Dict[str, Dict],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """Plot ROC curves for all models."""
    plt.figure(figsize=(8, 6))
    
    for name, res in results.items():
        if 'model' in res:
            y_pred_proba = res['model'].predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Test Set')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pr_curves(
    results: Dict[str, Dict],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """Plot Precision-Recall curves for all models."""
    plt.figure(figsize=(8, 6))
    
    # Baseline (no-skill) = positive class prevalence
    baseline = y_test.mean()
    plt.axhline(y=baseline, color='k', linestyle='--', label=f'No Skill (AP = {baseline:.3f})')
    
    for name, res in results.items():
        if 'model' in res:
            y_pred_proba = res['model'].predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = average_precision_score(y_test, y_pred_proba)
            plt.plot(recall, precision, label=f'{name} (AP = {pr_auc:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - Test Set')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_calibration_curves(
    results: Dict[str, Dict],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """Plot calibration curves for all models."""
    plt.figure(figsize=(8, 6))
    
    for name, res in results.items():
        if 'model' in res:
            y_pred_proba = res['model'].predict_proba(X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
            plt.plot(prob_pred, prob_true, 's-', label=name)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curves - Test Set')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(
    results: Dict[str, Dict],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: Optional[str] = None
) -> None:
    """Plot confusion matrices for all models."""
    n_models = len([r for r in results.values() if 'model' in r])
    if n_models == 0:
        return
    
    cols = min(2, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten()
    
    for idx, (name, res) in enumerate(results.items()):
        if 'model' not in res:
            continue
        ax = axes_flat[idx]
        y_pred = res['model'].predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{name}')
    
    # Hide unused subplots
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    if save_dir:
        save_path = os.path.join(save_dir, 'confusion_matrices.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metric_comparison(
    results: Dict[str, Dict],
    metrics: List[str] = None,
    save_path: Optional[str] = None
) -> None:
    """Plot bar chart comparing models on multiple metrics."""
    if metrics is None:
        metrics = ['cv_roc_auc_mean', 'cv_pr_auc_mean', 'cv_precision_mean', 'cv_recall_mean', 'cv_f1_mean']
    
    model_names = [name for name in results.keys() if 'model' in results[name]]
    n_models = len(model_names)
    n_metrics = len(metrics)
    
    x = np.arange(n_models)
    width = 0.8 / n_metrics
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        values = [results[name].get(metric, 0) for name in model_names]
        errors = [results[name].get(metric.replace('mean', 'std'), 0) for name in model_names]
        offset = (i - n_metrics/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=metric.replace('_mean', '').replace('cv_', '').upper(), yerr=errors, capsize=3)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison - Cross-Validation Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_test_metric_comparison(
    results: Dict[str, Dict],
    metrics: List[str] = None,
    save_path: Optional[str] = None
) -> None:
    """Plot bar chart comparing models on test set metrics."""
    if metrics is None:
        metrics = ['test_roc_auc', 'test_pr_auc', 'test_precision', 'test_recall', 'test_f1']
    
    model_names = [name for name in results.keys() if 'model' in results[name]]
    n_models = len(model_names)
    n_metrics = len(metrics)
    
    x = np.arange(n_models)
    width = 0.8 / n_metrics
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metrics):
        values = [results[name].get(metric, 0) for name in model_names]
        offset = (i - n_metrics/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=metric.replace('test_', '').upper())
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison - Test Set Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_model_comparison_report(
    results: Dict[str, Dict],
    save_path: str
) -> pd.DataFrame:
    """
    Generate comprehensive model comparison report.
    
    Args:
        results: Results dictionary from training
        save_path: Path to save CSV report
        
    Returns:
        Comparison DataFrame
    """
    comparison_data = []
    
    for name, res in results.items():
        if 'model' not in res:
            continue
        row = {'model': name}
        
        # CV metrics
        for key in ['cv_roc_auc_mean', 'cv_roc_auc_std', 'cv_pr_auc_mean', 'cv_pr_auc_std',
                    'cv_precision_mean', 'cv_precision_std', 'cv_recall_mean', 'cv_recall_std',
                    'cv_f1_mean', 'cv_f1_std']:
            if key in res:
                row[key] = res[key]
        
        # Test metrics
        for key in ['test_roc_auc', 'test_pr_auc', 'test_precision', 'test_recall', 'test_f1',
                    'test_tn', 'test_fp', 'test_fn', 'test_tp']:
            if key in res:
                row[key] = res[key]
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by test ROC-AUC descending
    if 'test_roc_auc' in df.columns:
        df = df.sort_values('test_roc_auc', ascending=False)
    
    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Model comparison report saved to {save_path}")
    
    return df


def print_model_summary(results: Dict[str, Dict]) -> None:
    """Print formatted model comparison summary."""
    print("\n" + "=" * 100)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 100)
    
    header = f"{'Model':<25} {'CV ROC-AUC':>12} {'CV PR-AUC':>12} {'CV F1':>10} {'Test ROC-AUC':>14} {'Test PR-AUC':>14} {'Test F1':>10}"
    print(header)
    print("-" * len(header))
    
    for name, res in sorted(results.items(), key=lambda x: x[1].get('test_roc_auc', 0), reverse=True):
        if 'model' not in res:
            continue
        cv_roc = f"{res.get('cv_roc_auc_mean', 0):.4f} (+/-{res.get('cv_roc_auc_std', 0):.4f})"
        cv_pr = f"{res.get('cv_pr_auc_mean', 0):.4f} (+/-{res.get('cv_pr_auc_std', 0):.4f})"
        cv_f1 = f"{res.get('cv_f1_mean', 0):.4f} (+/-{res.get('cv_f1_std', 0):.4f})"
        test_roc = f"{res.get('test_roc_auc', 0):.4f}"
        test_pr = f"{res.get('test_pr_auc', 0):.4f}"
        test_f1 = f"{res.get('test_f1', 0):.4f}"
        
        print(f"{name:<25} {cv_roc:>12} {cv_pr:>12} {cv_f1:>10} {test_roc:>14} {test_pr:>14} {test_f1:>10}")


def evaluate_all_models(
    results: Dict[str, Dict],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Run complete evaluation for all models and generate plots/report.
    
    Args:
        results: Results dictionary from training
        X_test: Test features
        y_test: Test targets
        save_dir: Directory to save outputs
        
    Returns:
        Comparison DataFrame
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        figures_dir = os.path.join(save_dir, 'figures')
        reports_dir = os.path.join(save_dir, 'reports')
        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
    else:
        figures_dir = reports_dir = None
    
    print_model_summary(results)
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    
    if figures_dir:
        plot_roc_curves(results, X_test, y_test, 
                       save_path=os.path.join(figures_dir, 'roc_curves.png'))
        plot_pr_curves(results, X_test, y_test,
                      save_path=os.path.join(figures_dir, 'pr_curves.png'))
        plot_calibration_curves(results, X_test, y_test,
                               save_path=os.path.join(figures_dir, 'calibration_curves.png'))
        plot_confusion_matrices(results, X_test, y_test,
                               save_dir=figures_dir)
        plot_metric_comparison(results,
                              save_path=os.path.join(figures_dir, 'cv_metric_comparison.png'))
        plot_test_metric_comparison(results,
                                   save_path=os.path.join(figures_dir, 'test_metric_comparison.png'))
    
    # Generate report
    if reports_dir:
        report_path = os.path.join(reports_dir, 'model_comparison_report.csv')
        comparison_df = generate_model_comparison_report(results, report_path)
    else:
        comparison_df = generate_model_comparison_report(results, 'model_comparison_report.csv')
    
    print("\nEvaluation complete!")
    return comparison_df


if __name__ == "__main__":
    pass