"""
Forecasting data loader and chronological split for WHO GHO India diabetes prevalence.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import os


def load_india_who_series(
    filepath: str,
    country_code: str = 'IND',
    sex: str = 'SEX_BTSX',
    age_group: str = 'AGEGROUP_YEARS18-PLUS'
) -> pd.Series:
    """
    Load and filter WHO GHO data for India national both-sexes 18+ series.
    
    Args:
        filepath: Path to WHO GHO CSV
        country_code: ISO country code (default: IND)
        sex: Sex dimension (default: SEX_BTSX for both sexes)
        age_group: Age group dimension (default: AGEGROUP_YEARS18-PLUS)
        
    Returns:
        Time series with year index and prevalence values
    """
    df = pd.read_csv(filepath)
    
    # Filter for India, both sexes, age 18+
    mask = (
        (df['SpatialDim'] == country_code) &
        (df['Dim1'] == sex) &
        (df['Dim2'] == age_group)
    )
    india_df = df[mask].copy()
    
    # Sort by year
    india_df = india_df.sort_values('TimeDim')
    
    # Create time series
    ts = pd.Series(
        india_df['NumericValue'].values,
        index=pd.Index(india_df['TimeDim'].values, name='year'),
        name='prevalence_pct'
    )
    
    # Also create uncertainty series
    low_ts = pd.Series(
        india_df['Low'].values,
        index=ts.index,
        name='low_ci'
    )
    high_ts = pd.Series(
        india_df['High'].values,
        index=ts.index,
        name='high_ci'
    )
    
    return ts, low_ts, high_ts, india_df


def chronological_train_val_test_split(
    ts: pd.Series,
    train_end_year: int = 2007,
    val_end_year: int = 2015,
    test_end_year: int = 2022
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Split time series chronologically into train/validation/test.
    
    With 33 years (1990-2022):
    - Train: 1990-2007 (18 years) - model fitting
    - Validation: 2008-2015 (8 years) - hyperparameter tuning/model selection
    - Test: 2016-2022 (7 years) - final held-out evaluation
    
    Args:
        ts: Full time series
        train_end_year: Last year of training period
        val_end_year: Last year of validation period
        test_end_year: Last year of test period (should be max year)
        
    Returns:
        Tuple of (train_ts, val_ts, test_ts)
    """
    train_ts = ts[ts.index <= train_end_year].copy()
    val_ts = ts[(ts.index > train_end_year) & (ts.index <= val_end_year)].copy()
    test_ts = ts[(ts.index > val_end_year) & (ts.index <= test_end_year)].copy()
    
    return train_ts, val_ts, test_ts


def print_split_info(train_ts: pd.Series, val_ts: pd.Series, test_ts: pd.Series) -> None:
    """Print information about the train/validation/test split."""
    print("=" * 60)
    print("CHRONOLOGICAL TRAIN/VALIDATION/TEST SPLIT")
    print("=" * 60)
    print(f"Train:     {train_ts.index.min()}–{train_ts.index.max()}  ({len(train_ts)} observations)")
    print(f"Validation: {val_ts.index.min()}–{val_ts.index.max()}  ({len(val_ts)} observations)")
    print(f"Test:      {test_ts.index.min()}–{test_ts.index.max()}  ({len(test_ts)} observations)")
    print(f"Total:     {ts.index.min()}–{ts.index.max()}  ({len(train_ts) + len(val_ts) + len(test_ts)} observations)")
    print()
    print("Train values:")
    for yr, val in train_ts.items():
        print(f"  {yr}: {val:.2f}%")
    print()
    print("Validation values:")
    for yr, val in val_ts.items():
        print(f"  {yr}: {val:.2f}%")
    print()
    print("Test values:")
    for yr, val in test_ts.items():
        print(f"  {yr}: {val:.2f}%")
    print("=" * 60)


def save_forecasting_data(
    train_ts: pd.Series,
    val_ts: pd.Series,
    test_ts: pd.Series,
    low_ts: pd.Series,
    high_ts: pd.Series,
    output_dir: str = 'data/processed'
) -> None:
    """Save processed forecasting data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all with split labels
    all_data = []
    for yr, val in train_ts.items():
        all_data.append({
            'year': yr,
            'prevalence_pct': val,
            'low_ci': low_ts.loc[yr],
            'high_ci': high_ts.loc[yr],
            'split': 'train'
        })
    for yr, val in val_ts.items():
        all_data.append({
            'year': yr,
            'prevalence_pct': val,
            'low_ci': low_ts.loc[yr],
            'high_ci': high_ts.loc[yr],
            'split': 'validation'
        })
    for yr, val in test_ts.items():
        all_data.append({
            'year': yr,
            'prevalence_pct': val,
            'low_ci': low_ts.loc[yr],
            'high_ci': high_ts.loc[yr],
            'split': 'test'
        })
    
    df_out = pd.DataFrame(all_data)
    output_path = os.path.join(output_dir, 'india_forecasting_data.csv')
    df_out.to_csv(output_path, index=False)
    print(f"Saved forecasting data to {output_path}")


if __name__ == "__main__":
    # Load India series
    filepath = r'C:\Users\ALWIN ABHISHEK\DiabPredict-AI\data\raw\who_diabetes_prevalence_agestd.csv'
    
    print("Loading India WHO GHO diabetes prevalence series...")
    ts, low_ts, high_ts, raw_df = load_india_who_series(filepath)
    
    print(f"\nLoaded series: {len(ts)} observations ({ts.index.min()}–{ts.index.max()})")
    print(f"Prevalence range: {ts.min():.2f}% – {ts.max():.2f}%")
    
    # Chronological split
    train_ts, val_ts, test_ts = chronological_train_val_test_split(ts)
    
    # Print split info
    print_split_info(train_ts, val_ts, test_ts)
    
    # Save processed data
    save_forecasting_data(train_ts, val_ts, test_ts, low_ts, high_ts)
    
    print("\nSTEP 1 COMPLETE: Data loaded and split chronologically.")