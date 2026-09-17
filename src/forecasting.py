"""
Time-series forecasting for population diabetes prevalence.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import warnings
import os
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False


def load_forecasting_data(filepath: str) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load processed forecasting data and return split time series.
    
    Args:
        filepath: Path to processed CSV
        
    Returns:
        Tuple of (train_ts, val_ts, test_ts, low_ts, high_ts)
    """
    df = pd.read_csv(filepath)
    
    train_df = df[df['split'] == 'train'].sort_values('year')
    val_df = df[df['split'] == 'validation'].sort_values('year')
    test_df = df[df['split'] == 'test'].sort_values('year')
    
    # Create datetime index for proper time series handling
    train_idx = pd.to_datetime(train_df['year'].astype(str) + '-01-01')
    val_idx = pd.to_datetime(val_df['year'].astype(str) + '-01-01')
    test_idx = pd.to_datetime(test_df['year'].astype(str) + '-01-01')
    
    train_ts = pd.Series(train_df['prevalence_pct'].values, index=train_idx, name='prevalence_pct')
    val_ts = pd.Series(val_df['prevalence_pct'].values, index=val_idx, name='prevalence_pct')
    test_ts = pd.Series(test_df['prevalence_pct'].values, index=test_idx, name='prevalence_pct')
    
    # For uncertainty series, use all years
    all_idx = pd.to_datetime(df['year'].astype(str) + '-01-01')
    low_ts = pd.Series(df['low_ci'].values, index=all_idx, name='low_ci')
    high_ts = pd.Series(df['high_ci'].values, index=all_idx, name='high_ci')
    
    return train_ts, val_ts, test_ts, low_ts, high_ts


def naive_forecast(train_ts: pd.Series, horizon: int) -> np.ndarray:
    """
    Naive forecast (last value carried forward).
    
    Args:
        train_ts: Training time series
        horizon: Number of steps to forecast
        
    Returns:
        Forecast array
    """
    last_value = train_ts.iloc[-1]
    return np.full(horizon, last_value)


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate forecast accuracy.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape)
    }


def linear_trend_forecast(train_ts: pd.Series, horizon: int) -> np.ndarray:
    """
    Linear regression with time feature.
    
    Args:
        train_ts: Training time series
        horizon: Number of steps to forecast
        
    Returns:
        Forecast array
    """
    # Create time feature (year index)
    t = np.arange(len(train_ts)).reshape(-1, 1)
    y = train_ts.values
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(t, y)
    
    # Forecast
    t_future = np.arange(len(train_ts), len(train_ts) + horizon).reshape(-1, 1)
    forecast = model.predict(t_future)
    
    return forecast


def select_arima_order(train_ts: pd.Series, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> Tuple[int, int, int]:
    """
    Select ARIMA order using AIC on training data.
    
    Args:
        train_ts: Training time series
        max_p: Maximum p value
        max_d: Maximum d value
        max_q: Maximum q value
        
    Returns:
        Best (p, d, q) order
    """
    if not ARIMA_AVAILABLE:
        return (1, 1, 1)
    
    best_aic = np.inf
    best_order = (1, 1, 1)
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and d == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(train_ts, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    
    print(f"  Selected ARIMA order: {best_order} (AIC={best_aic:.2f})")
    return best_order


def arima_forecast(train_ts: pd.Series, horizon: int, order: Optional[Tuple[int, int, int]] = None) -> Tuple[np.ndarray, Dict]:
    """
    ARIMA forecast with optional order selection.
    
    Args:
        train_ts: Training time series
        horizon: Number of steps to forecast
        order: ARIMA (p, d, q) order (None for auto-selection)
        
    Returns:
        Tuple of (forecast array, model info dict)
    """
    if not ARIMA_AVAILABLE:
        raise ImportError("statsmodels not available")
    
    if order is None:
        order = select_arima_order(train_ts)
    
    model = ARIMA(train_ts, order=order)
    fitted = model.fit()
    
    forecast_result = fitted.forecast(steps=horizon)
    forecast = forecast_result.values if hasattr(forecast_result, 'values') else np.array(forecast_result)
    
    # Get confidence intervals
    conf_int = fitted.get_forecast(steps=horizon).conf_int()
    
    info = {
        'order': order,
        'aic': fitted.aic,
        'bic': fitted.bic,
        'conf_int_lower': conf_int.iloc[:, 0].values,
        'conf_int_upper': conf_int.iloc[:, 1].values
    }
    
    return forecast, info


def xgboost_forecast(
    train_ts: pd.Series, 
    horizon: int, 
    n_lags: int = 3,
    n_estimators: int = 200,
    max_depth: int = 3,
    learning_rate: float = 0.1
) -> Tuple[np.ndarray, Dict]:
    """
    XGBoost forecast with lag features.
    
    Args:
        train_ts: Training time series
        horizon: Number of steps to forecast
        n_lags: Number of lag features to create
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        
    Returns:
        Tuple of (forecast array, model info dict)
    """
    # Create lag features
    df = pd.DataFrame({'y': train_ts.values})
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    
    df = df.dropna()
    X = df.drop('y', axis=1)
    y = df['y']
    
    from xgboost import XGBRegressor
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Recursive forecasting
    last_lags = df.drop('y', axis=1).iloc[-1:].values
    forecasts = []
    
    for _ in range(horizon):
        pred = model.predict(last_lags)[0]
        forecasts.append(pred)
        # Update lags for next prediction
        last_lags = np.roll(last_lags, -1)
        last_lags[0, -1] = pred
    
    info = {
        'n_lags': n_lags,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate
    }
    
    return np.array(forecasts), info


def run_xgboost_baseline(
    data_path: str = 'data/processed/india_forecasting_data.csv',
    output_dir: str = 'outputs'
) -> Dict:
    """
    Run XGBoost forecast baseline on validation period.
    
    Args:
        data_path: Path to processed forecasting data
        output_dir: Output directory for results
        
    Returns:
        Dictionary with results
    """
    print("=" * 60)
    print("XGBOOST FORECAST BASELINE")
    print("=" * 60)
    
    # Load data
    train_ts, val_ts, test_ts, low_ts, high_ts = load_forecasting_data(data_path)
    
    print(f"Train period: {train_ts.index.min()}–{train_ts.index.max()} ({len(train_ts)} obs)")
    print(f"Validation period: {val_ts.index.min()}–{val_ts.index.max()} ({len(val_ts)} obs)")
    print(f"Test period: {test_ts.index.min()}–{test_ts.index.max()} ({len(test_ts)} obs)")
    
    # XGBoost forecast on validation period
    horizon = len(val_ts)
    print(f"\nForecasting {horizon} steps ahead (validation period)...")
    
    xgb_pred, xgb_info = xgboost_forecast(train_ts, horizon)
    
    # Evaluate
    metrics = evaluate_forecast(val_ts.values, xgb_pred)
    
    print(f"\nValidation Metrics:")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  n_lags: {xgb_info['n_lags']}")
    print(f"  n_estimators: {xgb_info['n_estimators']}")
    
    # Print actual vs predicted
    print(f"\nActual vs Predicted (Validation):")
    print(f"{'Year':<6} {'Actual':<10} {'Predicted':<12} {'Error':<10}")
    print("-" * 40)
    for i, year in enumerate(val_ts.index):
        actual = val_ts.iloc[i]
        pred = xgb_pred[i]
        error = pred - actual
        print(f"{year:<6} {actual:<10.2f} {pred:<12.2f} {error:<10.2f}")
    
    # Save results
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # Save metrics
    results_df = pd.DataFrame([{
        'model': 'xgboost',
        'period': 'validation',
        'mae': metrics['mae'],
        'rmse': metrics['rmse'],
        'mape': metrics['mape'],
        'horizon': horizon,
        'n_lags': xgb_info['n_lags'],
        'n_estimators': xgb_info['n_estimators'],
        'max_depth': xgb_info['max_depth'],
        'learning_rate': xgb_info['learning_rate']
    }])
    metrics_path = os.path.join(output_dir, 'reports', 'xgboost_validation_metrics.csv')
    results_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Save actual vs predicted
    pred_df = pd.DataFrame({
        'year': val_ts.index,
        'actual': val_ts.values,
        'predicted': xgb_pred,
        'error': xgb_pred - val_ts.values
    })
    pred_path = os.path.join(output_dir, 'reports', 'xgboost_validation_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.plot(train_ts.index, train_ts.values, 'k-', label='Training (1990–2007)', linewidth=2)
    
    # Plot validation actual
    plt.plot(val_ts.index, val_ts.values, 'b-', label='Validation Actual (2008–2015)', linewidth=2, marker='o')
    
    # Plot validation predicted
    plt.plot(val_ts.index, xgb_pred, 'c--', label=f'XGBoost Forecast (lags={xgb_info["n_lags"]})', linewidth=2, marker='s')
    
    plt.xlabel('Year')
    plt.ylabel('Diabetes Prevalence (%)')
    plt.title('XGBoost Forecast Baseline - Validation Period (2008–2015)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'figures', 'xgboost_validation_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {plot_path}")
    
    print("=" * 60)
    print("XGBOOST BASELINE COMPLETE")
    print("=" * 60)
    
    return {
        'metrics': metrics,
        'predictions': xgb_pred,
        'info': xgb_info,
        'train_ts': train_ts,
        'val_ts': val_ts,
        'test_ts': test_ts
    }


def run_arima_baseline(
    data_path: str = 'data/processed/india_forecasting_data.csv',
    output_dir: str = 'outputs'
) -> Dict:
    """
    Run ARIMA forecast baseline on validation period.
    
    Args:
        data_path: Path to processed forecasting data
        output_dir: Output directory for results
        
    Returns:
        Dictionary with results
    """
    print("=" * 60)
    print("ARIMA FORECAST BASELINE")
    print("=" * 60)
    
    if not ARIMA_AVAILABLE:
        print("ERROR: statsmodels not available for ARIMA")
        return {'error': 'statsmodels not available'}
    
    # Load data
    train_ts, val_ts, test_ts, low_ts, high_ts = load_forecasting_data(data_path)
    
    print(f"Train period: {train_ts.index.min()}–{train_ts.index.max()} ({len(train_ts)} obs)")
    print(f"Validation period: {val_ts.index.min()}–{val_ts.index.max()} ({len(val_ts)} obs)")
    print(f"Test period: {test_ts.index.min()}–{test_ts.index.max()} ({len(test_ts)} obs)")
    
    # ARIMA forecast on validation period
    horizon = len(val_ts)
    print(f"\nForecasting {horizon} steps ahead (validation period)...")
    
    arima_pred, arima_info = arima_forecast(train_ts, horizon)
    
    # Evaluate
    metrics = evaluate_forecast(val_ts.values, arima_pred)
    
    print(f"\nValidation Metrics:")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  Order: {arima_info['order']}")
    print(f"  AIC: {arima_info['aic']:.2f}")
    print(f"  BIC: {arima_info['bic']:.2f}")
    
    # Print actual vs predicted
    print(f"\nActual vs Predicted (Validation):")
    print(f"{'Year':<6} {'Actual':<10} {'Predicted':<12} {'Error':<10}")
    print("-" * 40)
    for i, year in enumerate(val_ts.index):
        actual = val_ts.iloc[i]
        pred = arima_pred[i]
        error = pred - actual
        print(f"{year:<6} {actual:<10.2f} {pred:<12.2f} {error:<10.2f}")
    
    # Save results
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # Save metrics
    results_df = pd.DataFrame([{
        'model': 'arima',
        'period': 'validation',
        'mae': metrics['mae'],
        'rmse': metrics['rmse'],
        'mape': metrics['mape'],
        'horizon': horizon,
        'order_p': arima_info['order'][0],
        'order_d': arima_info['order'][1],
        'order_q': arima_info['order'][2],
        'aic': arima_info['aic'],
        'bic': arima_info['bic']
    }])
    metrics_path = os.path.join(output_dir, 'reports', 'arima_validation_metrics.csv')
    results_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Save actual vs predicted with confidence intervals
    pred_df = pd.DataFrame({
        'year': val_ts.index,
        'actual': val_ts.values,
        'predicted': arima_pred,
        'error': arima_pred - val_ts.values,
        'ci_lower': arima_info['conf_int_lower'],
        'ci_upper': arima_info['conf_int_upper']
    })
    pred_path = os.path.join(output_dir, 'reports', 'arima_validation_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.plot(train_ts.index, train_ts.values, 'k-', label='Training (1990–2007)', linewidth=2)
    
    # Plot validation actual
    plt.plot(val_ts.index, val_ts.values, 'b-', label='Validation Actual (2008–2015)', linewidth=2, marker='o')
    
    # Plot validation predicted
    plt.plot(val_ts.index, arima_pred, 'm--', label=f'ARIMA{arima_info["order"]} Forecast', linewidth=2, marker='s')
    
    # Plot confidence intervals
    plt.fill_between(val_ts.index, 
                     arima_info['conf_int_lower'], 
                     arima_info['conf_int_upper'],
                     color='m', alpha=0.2, label='95% CI')
    
    plt.xlabel('Year')
    plt.ylabel('Diabetes Prevalence (%)')
    plt.title(f'ARIMA{arima_info["order"]} Forecast - Validation Period (2008–2015)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'figures', 'arima_validation_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {plot_path}")
    
    print("=" * 60)
    print("ARIMA BASELINE COMPLETE")
    print("=" * 60)
    
    return {
        'metrics': metrics,
        'predictions': arima_pred,
        'info': arima_info,
        'train_ts': train_ts,
        'val_ts': val_ts,
        'test_ts': test_ts
    }


def run_linear_trend_baseline(
    data_path: str = 'data/processed/india_forecasting_data.csv',
    output_dir: str = 'outputs'
) -> Dict:
    """
    Run linear trend forecast baseline on validation period.
    
    Args:
        data_path: Path to processed forecasting data
        output_dir: Output directory for results
        
    Returns:
        Dictionary with results
    """
    print("=" * 60)
    print("LINEAR TREND FORECAST BASELINE")
    print("=" * 60)
    
    # Load data
    train_ts, val_ts, test_ts, low_ts, high_ts = load_forecasting_data(data_path)
    
    print(f"Train period: {train_ts.index.min()}–{train_ts.index.max()} ({len(train_ts)} obs)")
    print(f"Validation period: {val_ts.index.min()}–{val_ts.index.max()} ({len(val_ts)} obs)")
    print(f"Test period: {test_ts.index.min()}–{test_ts.index.max()} ({len(test_ts)} obs)")
    
    # Linear trend forecast on validation period
    horizon = len(val_ts)
    print(f"\nForecasting {horizon} steps ahead (validation period)...")
    
    linear_pred = linear_trend_forecast(train_ts, horizon)
    
    # Evaluate
    metrics = evaluate_forecast(val_ts.values, linear_pred)
    
    print(f"\nValidation Metrics:")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    
    # Print actual vs predicted
    print(f"\nActual vs Predicted (Validation):")
    print(f"{'Year':<6} {'Actual':<10} {'Predicted':<12} {'Error':<10}")
    print("-" * 40)
    for i, year in enumerate(val_ts.index):
        actual = val_ts.iloc[i]
        pred = linear_pred[i]
        error = pred - actual
        print(f"{year:<6} {actual:<10.2f} {pred:<12.2f} {error:<10.2f}")
    
    # Save results
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # Save metrics
    results_df = pd.DataFrame([{
        'model': 'linear_trend',
        'period': 'validation',
        'mae': metrics['mae'],
        'rmse': metrics['rmse'],
        'mape': metrics['mape'],
        'horizon': horizon,
        'slope': float(np.polyfit(np.arange(len(train_ts)), train_ts.values, 1)[0])
    }])
    metrics_path = os.path.join(output_dir, 'reports', 'linear_validation_metrics.csv')
    results_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Save actual vs predicted
    pred_df = pd.DataFrame({
        'year': val_ts.index,
        'actual': val_ts.values,
        'predicted': linear_pred,
        'error': linear_pred - val_ts.values
    })
    pred_path = os.path.join(output_dir, 'reports', 'linear_validation_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.plot(train_ts.index, train_ts.values, 'k-', label='Training (1990–2007)', linewidth=2)
    
    # Plot validation actual
    plt.plot(val_ts.index, val_ts.values, 'b-', label='Validation Actual (2008–2015)', linewidth=2, marker='o')
    
    # Plot validation predicted
    plt.plot(val_ts.index, linear_pred, 'g--', label='Linear Trend Forecast', linewidth=2, marker='s')
    
    # Plot linear trend line extended
    all_years = np.concatenate([train_ts.index, val_ts.index])
    t_all = np.arange(len(all_years))
    # Fit on training only for the line
    t_train = np.arange(len(train_ts))
    coeffs = np.polyfit(t_train, train_ts.values, 1)
    trend_line = np.polyval(coeffs, t_all)
    plt.plot(all_years, trend_line, 'g:', alpha=0.5, label=f'Linear Trend (slope={coeffs[0]:.3f}%/yr)')
    
    plt.xlabel('Year')
    plt.ylabel('Diabetes Prevalence (%)')
    plt.title('Linear Trend Forecast Baseline - Validation Period (2008–2015)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'figures', 'linear_validation_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {plot_path}")
    
    print("=" * 60)
    print("LINEAR TREND BASELINE COMPLETE")
    print("=" * 60)
    
    return {
        'metrics': metrics,
        'predictions': linear_pred,
        'train_ts': train_ts,
        'val_ts': val_ts,
        'test_ts': test_ts
    }


def run_naive_baseline(
    data_path: str = 'data/processed/india_forecasting_data.csv',
    output_dir: str = 'outputs'
) -> Dict:
    """
    Run naive forecast baseline on validation period.
    
    Args:
        data_path: Path to processed forecasting data
        output_dir: Output directory for results
        
    Returns:
        Dictionary with results
    """
    print("=" * 60)
    print("NAIVE FORECAST BASELINE")
    print("=" * 60)
    
    # Load data
    train_ts, val_ts, test_ts, low_ts, high_ts = load_forecasting_data(data_path)
    
    print(f"Train period: {train_ts.index.min()}–{train_ts.index.max()} ({len(train_ts)} obs)")
    print(f"Validation period: {val_ts.index.min()}–{val_ts.index.max()} ({len(val_ts)} obs)")
    print(f"Test period: {test_ts.index.min()}–{test_ts.index.max()} ({len(test_ts)} obs)")
    
    # Naive forecast on validation period
    horizon = len(val_ts)
    print(f"\nForecasting {horizon} steps ahead (validation period)...")
    
    naive_pred = naive_forecast(train_ts, horizon)
    
    # Evaluate
    metrics = evaluate_forecast(val_ts.values, naive_pred)
    
    print(f"\nValidation Metrics:")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    
    # Print actual vs predicted
    print(f"\nActual vs Predicted (Validation):")
    print(f"{'Year':<6} {'Actual':<10} {'Predicted':<12} {'Error':<10}")
    print("-" * 40)
    for i, year in enumerate(val_ts.index):
        actual = val_ts.iloc[i]
        pred = naive_pred[i]
        error = pred - actual
        print(f"{year:<6} {actual:<10.2f} {pred:<12.2f} {error:<10.2f}")
    
    # Save results
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # Save metrics
    results_df = pd.DataFrame([{
        'model': 'naive',
        'period': 'validation',
        'mae': metrics['mae'],
        'rmse': metrics['rmse'],
        'mape': metrics['mape'],
        'horizon': horizon,
        'last_train_value': train_ts.iloc[-1]
    }])
    metrics_path = os.path.join(output_dir, 'reports', 'naive_validation_metrics.csv')
    results_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Save actual vs predicted
    pred_df = pd.DataFrame({
        'year': val_ts.index,
        'actual': val_ts.values,
        'predicted': naive_pred,
        'error': naive_pred - val_ts.values
    })
    pred_path = os.path.join(output_dir, 'reports', 'naive_validation_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    
    # Plot training data
    plt.plot(train_ts.index, train_ts.values, 'k-', label='Training (1990–2007)', linewidth=2)
    
    # Plot validation actual
    plt.plot(val_ts.index, val_ts.values, 'b-', label='Validation Actual (2008–2015)', linewidth=2, marker='o')
    
    # Plot validation predicted
    plt.plot(val_ts.index, naive_pred, 'r--', label='Naive Forecast (Last Train Value)', linewidth=2, marker='s')
    
    # Add horizontal line for naive forecast value
    plt.axhline(y=train_ts.iloc[-1], color='r', linestyle=':', alpha=0.5, label=f'Naive Constant = {train_ts.iloc[-1]:.2f}%')
    
    plt.xlabel('Year')
    plt.ylabel('Diabetes Prevalence (%)')
    plt.title('Naive Forecast Baseline - Validation Period (2008–2015)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'figures', 'naive_validation_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {plot_path}")
    
    print("=" * 60)
    print("NAIVE BASELINE COMPLETE")
    print("=" * 60)
    
    return {
        'metrics': metrics,
        'predictions': naive_pred,
        'train_ts': train_ts,
        'val_ts': val_ts,
        'test_ts': test_ts
    }


def compare_all_models(data_path: str = 'data/processed/india_forecasting_data.csv',
                       output_dir: str = 'outputs') -> pd.DataFrame:
    """
    Compare all models on validation set and create comparison report.
    
    Args:
        data_path: Path to processed forecasting data
        output_dir: Output directory for results
        
    Returns:
        Comparison DataFrame
    """
    print("=" * 60)
    print("MODEL COMPARISON ON VALIDATION SET")
    print("=" * 60)
    
    # Load metrics from all models
    reports_dir = os.path.join(output_dir, 'reports')
    
    model_files = {
        'naive': 'naive_validation_metrics.csv',
        'linear_trend': 'linear_validation_metrics.csv',
        'arima': 'arima_validation_metrics.csv',
        'xgboost': 'xgboost_validation_metrics.csv'
    }
    
    comparison_rows = []
    for model_name, filename in model_files.items():
        filepath = os.path.join(reports_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            row = df.iloc[0].to_dict()
            row['model'] = model_name
            comparison_rows.append(row)
    
    if not comparison_rows:
        print("No model metrics found!")
        return pd.DataFrame()
    
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values('mae')
    
    print(f"\n{'Model':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10}")
    print("-" * 50)
    for _, row in comparison_df.iterrows():
        print(f"{row['model']:<20} {row['mae']:<10.4f} {row['rmse']:<10.4f} {row['mape']:<10.2f}%")
    
    best_model = comparison_df.iloc[0]['model']
    print(f"\nBest model on validation: {best_model} (MAE={comparison_df.iloc[0]['mae']:.4f})")
    
    # Save comparison
    os.makedirs(reports_dir, exist_ok=True)
    comparison_path = os.path.join(reports_dir, 'model_comparison_validation.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved comparison to {comparison_path}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    models = comparison_df['model'].values
    mae_vals = comparison_df['mae'].values
    rmse_vals = comparison_df['rmse'].values
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, mae_vals, width, label='MAE', color='skyblue', edgecolor='black')
    plt.bar(x + width/2, rmse_vals, width, label='RMSE', color='lightcoral', edgecolor='black')
    
    plt.xlabel('Model')
    plt.ylabel('Error')
    plt.title('Model Comparison - Validation Set (2008–2015)')
    plt.xticks(x, models, rotation=15)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'figures', 'model_comparison_validation.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {plot_path}")
    
    print("=" * 60)
    print("MODEL COMPARISON COMPLETE")
    print("=" * 60)
    
    return comparison_df


def run_final_forecast(
    data_path: str = 'data/processed/india_forecasting_data.csv',
    output_dir: str = 'outputs',
    forecast_years: int = 10
) -> Dict:
    """
    Run final forecasting with the best model (ARIMA).
    
    1. Retrain on train+validation (1990–2015)
    2. Evaluate on held-out test (2016–2022)
    3. Forecast future years beyond 2022
    
    Args:
        data_path: Path to processed forecasting data
        output_dir: Output directory for results
        forecast_years: Number of years to forecast beyond 2022
        
    Returns:
        Dictionary with results
    """
    print("=" * 60)
    print("FINAL FORECASTING WITH BEST MODEL (ARIMA)")
    print("=" * 60)
    
    if not ARIMA_AVAILABLE:
        print("ERROR: statsmodels not available for ARIMA")
        return {'error': 'statsmodels not available'}
    
    # Load data
    train_ts, val_ts, test_ts, low_ts, high_ts = load_forecasting_data(data_path)
    
    print(f"Train period: {train_ts.index.min().year}–{train_ts.index.max().year} ({len(train_ts)} obs)")
    print(f"Validation period: {val_ts.index.min().year}–{val_ts.index.max().year} ({len(val_ts)} obs)")
    print(f"Test period: {test_ts.index.min().year}–{test_ts.index.max().year} ({len(test_ts)} obs)")
    
    # Combine train + validation for final model fitting
    train_val_ts = pd.concat([train_ts, val_ts])
    print(f"\nCombined train+val: {train_val_ts.index.min().year}–{train_val_ts.index.max().year} ({len(train_val_ts)} obs)")
    
    # Fit ARIMA with best order from validation (2, 2, 0)
    best_order = (2, 2, 0)
    print(f"\nFitting ARIMA{best_order} on train+validation...")
    
    model = ARIMA(train_val_ts, order=best_order)
    fitted = model.fit()
    print(f"AIC: {fitted.aic:.2f}, BIC: {fitted.bic:.2f}")
    
    # Step 1: Forecast test period (2016–2022) for held-out evaluation
    test_horizon = len(test_ts)
    print(f"\n1. Forecasting {test_horizon} steps for held-out test period...")
    
    test_forecast = fitted.forecast(steps=test_horizon)
    test_pred = test_forecast.values if hasattr(test_forecast, 'values') else np.array(test_forecast)
    
    test_conf_int = fitted.get_forecast(steps=test_horizon).conf_int()
    
    test_metrics = evaluate_forecast(test_ts.values, test_pred)
    
    print(f"\nTest Metrics (2016–2022):")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")
    
    print(f"\nActual vs Predicted (Test):")
    print(f"{'Year':<6} {'Actual':<10} {'Predicted':<12} {'Error':<10}")
    print("-" * 40)
    for i, year in enumerate(test_ts.index):
        actual = test_ts.iloc[i]
        pred = test_pred[i]
        error = pred - actual
        print(f"{year.year:<6} {actual:<10.2f} {pred:<12.2f} {error:<10.2f}")
    
    # Step 2: Forecast future years beyond 2022
    print(f"\n2. Forecasting {forecast_years} future years (2023–{2022+forecast_years})...")
    
    future_horizon = forecast_years
    future_forecast = fitted.forecast(steps=future_horizon)
    future_pred = future_forecast.values if hasattr(future_forecast, 'values') else np.array(future_forecast)
    
    future_conf_int = fitted.get_forecast(steps=future_horizon).conf_int()
    future_years = list(range(2023, 2023 + forecast_years))
    
    print(f"\nFuture Forecast:")
    print(f"{'Year':<6} {'Predicted':<12} {'Lower 95%':<12} {'Upper 95%':<12}")
    print("-" * 45)
    for i, year in enumerate(future_years):
        print(f"{year:<6} {future_pred[i]:<12.2f} {future_conf_int.iloc[i, 0]:<12.2f} {future_conf_int.iloc[i, 1]:<12.2f}")
    
    # Save results
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    # Save test evaluation
    test_results_df = pd.DataFrame([{
        'model': 'arima',
        'period': 'test',
        'mae': test_metrics['mae'],
        'rmse': test_metrics['rmse'],
        'mape': test_metrics['mape'],
        'horizon': test_horizon,
        'order': str(best_order)
    }])
    test_metrics_path = os.path.join(output_dir, 'reports', 'arima_test_metrics.csv')
    test_results_df.to_csv(test_metrics_path, index=False)
    print(f"\nSaved test metrics to {test_metrics_path}")
    
    # Save test predictions
    test_pred_df = pd.DataFrame({
        'year': test_ts.index.year,
        'actual': test_ts.values,
        'predicted': test_pred,
        'error': test_pred - test_ts.values,
        'ci_lower': test_conf_int.iloc[:, 0].values,
        'ci_upper': test_conf_int.iloc[:, 1].values
    })
    test_pred_path = os.path.join(output_dir, 'reports', 'arima_test_predictions.csv')
    test_pred_df.to_csv(test_pred_path, index=False)
    print(f"Saved test predictions to {test_pred_path}")
    
    # Save future forecast
    future_df = pd.DataFrame({
        'year': future_years,
        'predicted': future_pred,
        'ci_lower': future_conf_int.iloc[:, 0].values,
        'ci_upper': future_conf_int.iloc[:, 1].values
    })
    future_path = os.path.join(output_dir, 'reports', 'arima_future_forecast.csv')
    future_df.to_csv(future_path, index=False)
    print(f"Saved future forecast to {future_path}")
    
    # Plot 1: Historical + Test Evaluation
    plt.figure(figsize=(12, 6))
    
    plt.plot(train_ts.index, train_ts.values, 'k-', label='Training (1990–2007)', linewidth=2)
    plt.plot(val_ts.index, val_ts.values, 'b-', label='Validation (2008–2015)', linewidth=2, marker='o')
    plt.plot(test_ts.index, test_ts.values, 'g-', label='Test Actual (2016–2022)', linewidth=2, marker='o')
    plt.plot(test_ts.index, test_pred, 'r--', label='ARIMA Test Forecast', linewidth=2, marker='s')
    plt.fill_between(test_ts.index, 
                     test_conf_int.iloc[:, 0].values, 
                     test_conf_int.iloc[:, 1].values,
                     color='r', alpha=0.2, label='95% CI (Test)')
    
    plt.xlabel('Year')
    plt.ylabel('Diabetes Prevalence (%)')
    plt.title('ARIMA(2,2,0) - Held-Out Test Evaluation (2016–2022)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'figures', 'arima_test_evaluation.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved test evaluation plot to {plot_path}")
    
    # Plot 2: Complete historical + Future forecast
    plt.figure(figsize=(12, 6))
    
    # All historical data
    all_historical = pd.concat([train_ts, val_ts, test_ts])
    plt.plot(all_historical.index, all_historical.values, 'k-', label='Historical WHO Estimates (1990–2022)', linewidth=2, marker='o', markersize=4)
    
    # Future forecast
    future_idx = pd.to_datetime([f'{y}-01-01' for y in future_years])
    plt.plot(future_idx, future_pred, 'r--', label=f'ARIMA(2,2,0) Forecast (2023–{2022+forecast_years})', linewidth=2, marker='s')
    plt.fill_between(future_idx, 
                     future_conf_int.iloc[:, 0].values, 
                     future_conf_int.iloc[:, 1].values,
                     color='r', alpha=0.2, label='95% Prediction Interval')
    
    plt.xlabel('Year')
    plt.ylabel('Diabetes Prevalence (%)')
    plt.title(f'India Diabetes Prevalence: Historical Estimates + ARIMA Forecast (2023–{2022+forecast_years})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'figures', 'arima_future_forecast.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved future forecast plot to {plot_path}")
    
    # Plot 3: Actual vs Predicted on test
    plt.figure(figsize=(10, 6))
    plt.plot(test_ts.index.year, test_ts.values, 'bo-', label='Actual', linewidth=2, markersize=8)
    plt.plot(test_ts.index.year, test_pred, 'rs--', label='Predicted', linewidth=2, markersize=8)
    plt.fill_between(test_ts.index.year, 
                     test_conf_int.iloc[:, 0].values, 
                     test_conf_int.iloc[:, 1].values,
                     color='r', alpha=0.2, label='95% CI')
    
    plt.xlabel('Year')
    plt.ylabel('Diabetes Prevalence (%)')
    plt.title('ARIMA(2,2,0) - Actual vs Predicted on Test Set (2016–2022)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(test_ts.index.year)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'figures', 'arima_test_actual_vs_predicted.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved actual vs predicted test plot to {plot_path}")
    
    print("=" * 60)
    print("FINAL FORECASTING COMPLETE")
    print("=" * 60)
    
    return {
        'test_metrics': test_metrics,
        'test_predictions': test_pred,
        'test_conf_int': test_conf_int,
        'future_forecast': future_pred,
        'future_conf_int': future_conf_int,
        'future_years': future_years,
        'fitted_model': fitted
    }


if __name__ == "__main__":
    import sys
    data_path = r'C:\Users\ALWIN ABHISHEK\DiabPredict-AI\data\processed\india_forecasting_data.csv'
    output_dir = r'C:\Users\ALWIN ABHISHEK\DiabPredict-AI\outputs'
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'linear':
            run_linear_trend_baseline(data_path, output_dir)
        elif sys.argv[1] == 'arima':
            run_arima_baseline(data_path, output_dir)
        elif sys.argv[1] == 'xgboost':
            run_xgboost_baseline(data_path, output_dir)
        elif sys.argv[1] == 'compare':
            compare_all_models(data_path, output_dir)
        elif sys.argv[1] == 'final':
            run_final_forecast(data_path, output_dir)
        else:
            run_naive_baseline(data_path, output_dir)
    else:
        run_naive_baseline(data_path, output_dir)