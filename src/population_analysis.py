"""
Population Diabetes Trend and Risk-Factor Analysis Module.

This module provides functions to analyze diabetes prevalence across
demographic, lifestyle, and health-indicator groups using the CDC BRFSS 2015 dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# BRFSS 2015 variable mappings for interpretation
VARIABLE_LABELS = {
    'Diabetes_binary': 'Diabetes Status',
    'HighBP': 'High Blood Pressure',
    'HighChol': 'High Cholesterol',
    'CholCheck': 'Cholesterol Check in 5 Years',
    'BMI': 'Body Mass Index',
    'Smoker': 'Smoker',
    'Stroke': 'Stroke',
    'HeartDiseaseorAttack': 'Heart Disease or Attack',
    'PhysActivity': 'Physical Activity',
    'Fruits': 'Fruit Consumption',
    'Veggies': 'Vegetable Consumption',
    'HvyAlcoholConsump': 'Heavy Alcohol Consumption',
    'AnyHealthcare': 'Healthcare Coverage',
    'NoDocbcCost': 'Could Not Afford Doctor',
    'GenHlth': 'General Health',
    'MentHlth': 'Mental Health Days',
    'PhysHlth': 'Physical Health Days',
    'DiffWalk': 'Difficulty Walking',
    'Sex': 'Sex',
    'Age': 'Age Category',
    'Education': 'Education Level',
    'Income': 'Income Level'
}

# Value mappings for categorical variables
VALUE_LABELS = {
    'Diabetes_binary': {0: 'No Diabetes', 1: 'Diabetes'},
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
        1: 'Never attended/Kindergarten',
        2: 'Elementary',
        3: 'Some High School',
        4: 'High School Graduate',
        5: 'Some College/Technical',
        6: 'College Graduate'
    },
    'Income': {
        1: '<$10,000', 2: '$10,000-$15,000', 3: '$15,000-$20,000',
        4: '$20,000-$25,000', 5: '$25,000-$35,000', 6: '$35,000-$50,000',
        7: '$50,000-$75,000', 8: '$75,000+'
    }
}

# BMI categories
BMI_CATEGORIES = [
    (0, 18.5, 'Underweight'),
    (18.5, 25, 'Normal'),
    (25, 30, 'Overweight'),
    (30, 35, 'Obese Class I'),
    (35, 40, 'Obese Class II'),
    (40, 100, 'Obese Class III')
]


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load and clean the BRFSS dataset.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Cleaned dataframe
    """
    df = pd.read_csv(filepath)
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    print(f"Loaded {initial_rows} rows, removed {removed} duplicates ({removed/initial_rows*100:.1f}%)")
    print(f"Cleaned dataset shape: {df.shape}")
    return df


def calculate_overall_prevalence(df: pd.DataFrame, target_col: str = 'Diabetes_binary') -> Dict:
    """
    Calculate overall diabetes prevalence.
    
    Args:
        df: Cleaned dataframe
        target_col: Target column name
        
    Returns:
        Dictionary with prevalence statistics
    """
    total = len(df)
    cases = df[target_col].sum()
    prevalence = cases / total * 100
    
    result = {
        'total_sample': total,
        'diabetes_cases': int(cases),
        'prevalence_pct': prevalence
    }
    
    print(f"\n=== Overall Diabetes Prevalence ===")
    print(f"Total sample: {total:,}")
    print(f"Diabetes cases: {int(cases):,}")
    print(f"Prevalence: {prevalence:.2f}%")
    
    return result


def analyze_prevalence_by_feature(
    df: pd.DataFrame,
    feature: str,
    target_col: str = 'Diabetes_binary',
    feature_labels: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Calculate diabetes prevalence by a categorical/ordinal feature.
    
    Args:
        df: Cleaned dataframe
        feature: Feature column name
        target_col: Target column name
        feature_labels: Optional mapping of values to labels
        
    Returns:
        DataFrame with prevalence statistics
    """
    grouped = df.groupby(feature)[target_col].agg(['count', 'sum']).reset_index()
    grouped.columns = [feature, 'sample_count', 'diabetes_cases']
    grouped['prevalence_pct'] = (grouped['diabetes_cases'] / grouped['sample_count'] * 100).round(2)
    
    # Add labels if provided
    if feature_labels and feature in feature_labels:
        grouped[f'{feature}_label'] = grouped[feature].map(feature_labels)
    
    # Sort by feature value for ordinal features
    grouped = grouped.sort_values(feature).reset_index(drop=True)
    
    return grouped


def create_bmi_categories(df: pd.DataFrame, bmi_col: str = 'BMI') -> pd.Series:
    """
    Create BMI category labels.
    
    Args:
        df: Dataframe with BMI column
        bmi_col: BMI column name
        
    Returns:
        Series with BMI category labels
    """
    def categorize_bmi(bmi):
        for low, high, label in BMI_CATEGORIES:
            if low <= bmi < high:
                return label
        return 'Unknown'
    
    return df[bmi_col].apply(categorize_bmi)


def analyze_all_features(
    df: pd.DataFrame,
    target_col: str = 'Diabetes_binary'
) -> Dict[str, pd.DataFrame]:
    """
    Analyze prevalence by all specified features.
    
    Args:
        df: Cleaned dataframe
        target_col: Target column name
        
    Returns:
        Dictionary mapping feature names to prevalence DataFrames
    """
    results = {}
    
    # Binary features
    binary_features = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
                       'PhysActivity', 'DiffWalk', 'Sex']
    
    for feat in binary_features:
        if feat in df.columns:
            results[feat] = analyze_prevalence_by_feature(df, feat, target_col, VALUE_LABELS)
            print(f"\n=== Prevalence by {VARIABLE_LABELS.get(feat, feat)} ===")
            print(results[feat].to_string(index=False))
    
    # Ordinal features
    ordinal_features = ['GenHlth', 'Age', 'Education', 'Income']
    for feat in ordinal_features:
        if feat in df.columns:
            results[feat] = analyze_prevalence_by_feature(df, feat, target_col, VALUE_LABELS)
            print(f"\n=== Prevalence by {VARIABLE_LABELS.get(feat, feat)} ===")
            print(results[feat].to_string(index=False))
    
    # BMI categories
    if 'BMI' in df.columns:
        df['BMI_category'] = create_bmi_categories(df)
        results['BMI_category'] = analyze_prevalence_by_feature(df, 'BMI_category', target_col)
        print(f"\n=== Prevalence by BMI Category ===")
        print(results['BMI_category'].to_string(index=False))
    
    return results


def save_prevalence_tables(results: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """
    Save prevalence analysis tables to CSV files.
    
    Args:
        results: Dictionary of prevalence DataFrames
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual tables
    for feature, table in results.items():
        filepath = os.path.join(output_dir, f'prevalence_by_{feature}.csv')
        table.to_csv(filepath, index=False)
        print(f"Saved: {filepath}")
    
    # Save combined summary
    combined_rows = []
    for feature, table in results.items():
        for _, row in table.iterrows():
            combined_rows.append({
                'feature': feature,
                'category': row.get(f'{feature}_label', row[feature]),
                'sample_count': row['sample_count'],
                'diabetes_cases': row['diabetes_cases'],
                'prevalence_pct': row['prevalence_pct']
            })
    
    combined_df = pd.DataFrame(combined_rows)
    combined_path = os.path.join(output_dir, 'prevalence_summary_all_features.csv')
    combined_df.to_csv(combined_path, index=False)
    print(f"Saved combined summary: {combined_path}")


def plot_prevalence_by_age(
    prevalence_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """Plot diabetes prevalence by age category."""
    plt.figure(figsize=(10, 6))
    
    x_labels = prevalence_df.get('Age_label', prevalence_df['Age'])
    plt.bar(range(len(prevalence_df)), prevalence_df['prevalence_pct'], 
            color='steelblue', edgecolor='black', alpha=0.8)
    
    plt.xlabel('Age Category')
    plt.ylabel('Diabetes Prevalence (%)')
    plt.title('Diabetes Prevalence by Age Category (BRFSS 2015)')
    plt.xticks(range(len(prevalence_df)), x_labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, v in enumerate(prevalence_df['prevalence_pct']):
        plt.text(i, v + 0.2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prevalence_by_bmi(
    prevalence_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """Plot diabetes prevalence by BMI category."""
    plt.figure(figsize=(10, 6))
    
    order = ['Underweight', 'Normal', 'Overweight', 'Obese Class I', 'Obese Class II', 'Obese Class III']
    df_plot = prevalence_df.set_index('BMI_category').reindex(order).reset_index()
    
    bars = plt.bar(range(len(df_plot)), df_plot['prevalence_pct'], 
            color=['lightcoral' if 'Obese' in str(x) else 'steelblue' for x in df_plot['BMI_category']], 
            edgecolor='black', alpha=0.8)
    
    plt.xlabel('BMI Category')
    plt.ylabel('Diabetes Prevalence (%)')
    plt.title('Diabetes Prevalence by BMI Category (BRFSS 2015)')
    plt.xticks(range(len(df_plot)), df_plot['BMI_category'], rotation=15, ha='right')
    
    for i, v in enumerate(df_plot['prevalence_pct']):
        if not pd.isna(v):
            plt.text(i, v + 0.3, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prevalence_by_gender(
    prevalence_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """Plot diabetes prevalence by gender."""
    plt.figure(figsize=(8, 6))
    
    labels = prevalence_df.get('Sex_label', prevalence_df['Sex'])
    colors = ['lightpink', 'lightblue']
    
    bars = plt.bar(range(len(prevalence_df)), prevalence_df['prevalence_pct'], 
            color=colors[:len(prevalence_df)], edgecolor='black', alpha=0.8)
    
    plt.xlabel('Sex')
    plt.ylabel('Diabetes Prevalence (%)')
    plt.title('Diabetes Prevalence by Sex (BRFSS 2015)')
    plt.xticks(range(len(prevalence_df)), labels)
    
    for i, v in enumerate(prevalence_df['prevalence_pct']):
        plt.text(i, v + 0.1, f'{v:.1f}%', ha='center', va='bottom', fontsize=11)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prevalence_by_genhlth(
    prevalence_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """Plot diabetes prevalence by general health."""
    plt.figure(figsize=(10, 6))
    
    labels = prevalence_df.get('GenHlth_label', prevalence_df['GenHlth'])
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(prevalence_df)))
    
    bars = plt.bar(range(len(prevalence_df)), prevalence_df['prevalence_pct'], 
            color=colors, edgecolor='black', alpha=0.8)
    
    plt.xlabel('General Health')
    plt.ylabel('Diabetes Prevalence (%)')
    plt.title('Diabetes Prevalence by General Health Status (BRFSS 2015)')
    plt.xticks(range(len(prevalence_df)), labels, rotation=15, ha='right')
    
    for i, v in enumerate(prevalence_df['prevalence_pct']):
        plt.text(i, v + 0.3, f'{v:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prevalence_by_physactivity(
    prevalence_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """Plot diabetes prevalence by physical activity."""
    plt.figure(figsize=(8, 6))
    
    labels = prevalence_df.get('PhysActivity_label', prevalence_df['PhysActivity'])
    colors = ['lightcoral', 'lightgreen']
    
    bars = plt.bar(range(len(prevalence_df)), prevalence_df['prevalence_pct'], 
            color=colors[:len(prevalence_df)], edgecolor='black', alpha=0.8)
    
    plt.xlabel('Physical Activity')
    plt.ylabel('Diabetes Prevalence (%)')
    plt.title('Diabetes Prevalence by Physical Activity (BRFSS 2015)')
    plt.xticks(range(len(prevalence_df)), labels)
    
    for i, v in enumerate(prevalence_df['prevalence_pct']):
        plt.text(i, v + 0.1, f'{v:.1f}%', ha='center', va='bottom', fontsize=11)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prevalence_by_income(
    prevalence_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """Plot diabetes prevalence by income level."""
    plt.figure(figsize=(12, 6))
    
    labels = prevalence_df.get('Income_label', prevalence_df['Income'])
    
    bars = plt.bar(range(len(prevalence_df)), prevalence_df['prevalence_pct'], 
            color='steelblue', edgecolor='black', alpha=0.8)
    
    plt.xlabel('Income Level')
    plt.ylabel('Diabetes Prevalence (%)')
    plt.title('Diabetes Prevalence by Income Level (BRFSS 2015)')
    plt.xticks(range(len(prevalence_df)), labels, rotation=45, ha='right')
    
    for i, v in enumerate(prevalence_df['prevalence_pct']):
        plt.text(i, v + 0.2, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_prevalence_heatmap(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = 'Diabetes_binary',
    save_path: Optional[str] = None
) -> None:
    """
    Create a heatmap of diabetes prevalence across feature combinations.
    
    Args:
        df: Cleaned dataframe
        features: List of features to cross-tabulate
        target_col: Target column
        save_path: Path to save figure
    """
    # Create a simplified heatmap for key binary features
    n_features = len(features)
    fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 5))
    
    if n_features == 1:
        axes = [axes]
    
    for idx, feat in enumerate(features):
        if feat not in df.columns:
            continue
            
        # Create cross-tabulation
        ct = pd.crosstab(df[feat], df[target_col], normalize='index') * 100
        
        # Plot heatmap
        sns.heatmap(ct, annot=True, fmt='.1f', cmap='Reds', 
                    cbar=idx == n_features-1, ax=axes[idx])
        
        axes[idx].set_xlabel('Diabetes')
        axes[idx].set_ylabel(VARIABLE_LABELS.get(feat, feat))
        axes[idx].set_title(f'Prevalence by {VARIABLE_LABELS.get(feat, feat)}')
        
        # Set tick labels
        if feat in VALUE_LABELS:
            y_labels = [VALUE_LABELS[feat].get(i, str(i)) for i in ct.index]
            x_labels = ['No Diabetes', 'Diabetes']
            axes[idx].set_yticklabels(y_labels, rotation=0)
            axes[idx].set_xticklabels(x_labels, rotation=0)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_association_measures(
    df: pd.DataFrame,
    target_col: str = 'Diabetes_binary',
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute association measures (phi coefficient, Cramer's V) between features and diabetes.
    
    Args:
        df: Cleaned dataframe
        target_col: Target column
        features: List of features to analyze
        
    Returns:
        DataFrame with association measures
    """
    from scipy.stats import chi2_contingency
    
    if features is None:
        features = ['HighBP', 'HighChol', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
                    'PhysActivity', 'DiffWalk', 'Sex', 'GenHlth', 'Age', 'Education', 'Income']
    
    results = []
    
    for feat in features:
        if feat not in df.columns:
            continue
            
        # Create contingency table
        ct = pd.crosstab(df[feat], df[target_col])
        
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2, p, dof, expected = chi2_contingency(ct)
            n = ct.sum().sum()
            
            # Phi coefficient (for 2x2) or Cramer's V
            min_dim = min(ct.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
            
            results.append({
                'feature': feat,
                'feature_label': VARIABLE_LABELS.get(feat, feat),
                'chi2': chi2,
                'p_value': p,
                'cramers_v': cramers_v,
                'dof': dof
            })
    
    return pd.DataFrame(results).sort_values('cramers_v', ascending=False)


def plot_association_measures(
    assoc_df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """Plot Cramer's V association measures."""
    plt.figure(figsize=(10, 8))
    
    # Filter out features with very small association
    plot_df = assoc_df[assoc_df['cramers_v'] > 0.01].copy()
    
    colors = plt.cm.Reds(plot_df['cramers_v'] / plot_df['cramers_v'].max())
    bars = plt.barh(range(len(plot_df)), plot_df['cramers_v'], color=colors, edgecolor='black')
    
    plt.yticks(range(len(plot_df)), plot_df['feature_label'])
    plt.xlabel("Cramer's V (Association Strength)")
    plt.title("Association Strength with Diabetes Status (Cramer's V)")
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(plot_df['cramers_v']):
        plt.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def run_population_analysis(
    data_path: str,
    output_dir: str = 'outputs'
) -> Dict:
    """
    Run complete population analysis pipeline.
    
    Args:
        data_path: Path to BRFSS CSV file
        output_dir: Output directory for results
        
    Returns:
        Dictionary with analysis results
    """
    print("=" * 70)
    print("POPULATION DIABETES TREND AND RISK-FACTOR ANALYSIS")
    print("CDC BRFSS 2015 Dataset")
    print("=" * 70)
    
    # Load and clean data
    print("\n1. LOADING AND CLEANING DATA...")
    df = load_and_clean_data(data_path)
    
    # Overall prevalence
    print("\n2. CALCULATING OVERALL PREVALENCE...")
    overall = calculate_overall_prevalence(df)
    
    # Analyze by features
    print("\n3. ANALYZING PREVALENCE BY FEATURES...")
    results = analyze_all_features(df)
    
    # Save tables
    print("\n4. SAVING PREVALENCE TABLES...")
    reports_dir = os.path.join(output_dir, 'reports')
    save_prevalence_tables(results, reports_dir)
    
    # Generate visualizations
    print("\n5. GENERATING VISUALIZATIONS...")
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Individual plots
    if 'Age' in results:
        plot_prevalence_by_age(results['Age'], 
                              os.path.join(figures_dir, 'prevalence_by_age.png'))
    
    if 'BMI_category' in results:
        plot_prevalence_by_bmi(results['BMI_category'],
                              os.path.join(figures_dir, 'prevalence_by_bmi.png'))
    
    if 'Sex' in results:
        plot_prevalence_by_gender(results['Sex'],
                                 os.path.join(figures_dir, 'prevalence_by_gender.png'))
    
    if 'GenHlth' in results:
        plot_prevalence_by_genhlth(results['GenHlth'],
                                  os.path.join(figures_dir, 'prevalence_by_genhlth.png'))
    
    if 'PhysActivity' in results:
        plot_prevalence_by_physactivity(results['PhysActivity'],
                                       os.path.join(figures_dir, 'prevalence_by_physactivity.png'))
    
    if 'Income' in results:
        plot_prevalence_by_income(results['Income'],
                                 os.path.join(figures_dir, 'prevalence_by_income.png'))
    
    # Association heatmap
    binary_features = ['HighBP', 'HighChol', 'Smoker', 'PhysActivity', 'DiffWalk', 'Sex']
    plot_prevalence_heatmap(df, binary_features,
                           save_path=os.path.join(figures_dir, 'prevalence_heatmap.png'))
    
    # Association measures
    print("\n6. COMPUTING ASSOCIATION MEASURES...")
    assoc_df = compute_association_measures(df)
    assoc_path = os.path.join(reports_dir, 'association_measures.csv')
    assoc_df.to_csv(assoc_path, index=False)
    print(f"Saved association measures: {assoc_path}")
    print("\nTop associations with diabetes:")
    print(assoc_df.head(10).to_string(index=False))
    
    plot_association_measures(assoc_df,
                             os.path.join(figures_dir, 'association_measures.png'))
    
    # Save overall prevalence
    overall_path = os.path.join(reports_dir, 'overall_prevalence.csv')
    pd.DataFrame([overall]).to_csv(overall_path, index=False)
    
    print("\n" + "=" * 70)
    print("POPULATION ANALYSIS COMPLETE")
    print("=" * 70)
    
    return {
        'overall_prevalence': overall,
        'prevalence_by_feature': results,
        'association_measures': assoc_df,
        'cleaned_data': df
    }


if __name__ == "__main__":
    data_path = r'C:\Users\ALWIN ABHISHEK\DiabPredict-AI\data\raw\diabetes_binary_health_indicators_BRFSS2015.csv'
    run_population_analysis(data_path)