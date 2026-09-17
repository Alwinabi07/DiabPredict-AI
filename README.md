# DiabPredict AI

A comprehensive diabetes risk prediction and population prevalence forecasting system built with Streamlit, scikit-learn, XGBoost, and statsmodels.

![DiabPredict AI Banner](https://img.shields.io/badge/Streamlit-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)

## Project Overview

DiabPredict AI is an end-to-end machine learning system for diabetes risk prediction and population prevalence forecasting. The application provides:

- **Module 1**: Individual diabetes risk prediction using 21 BRFSS (Behavioral Risk Factor Surveillance System) features
- **Module 2**: Population-level diabetes prevalence analysis using CDC BRFSS 2015 data
- **Module 3**: Prevalence forecasting using WHO data and ARIMA time-series modeling
- **Module 4**: Explainable AI (SHAP) explanations for the trained model
- **Module 5**: What-If simulation to explore how changing inputs affects predicted risk

The system is designed for **screening and research purposes only** and does not provide medical diagnoses or clinical decisions.

## Architecture

```
DiabPredict AI/
├── app/
│   └── app.py              # Main Streamlit application with navigation
├── src/
│   ├── predict.py          # DiabetesRiskPredictor wrapper & utilities
│   ├── explainability.py   # SHAP explainer and explanation logic
│   ├── explainability_page.py  # SHAP explanations page (bug fixes)
│   ├── what_if_simulation.py   # What-If simulation page (new)
│   ├── population_analytics.py   # Module 2: population analysis
│   ├── prevalence_forecast.py      # Module 3: prevalence forecasting
│   ├── population_analysis.py  # Data analysis module
│   ├── preprocessing.py    # BRFSS preprocessing pipeline
│   ├── evaluate.py         # Model evaluation utilities
│   ├── forecasting.py      # Forecasting utilities
│   ├── forecasting_data.py # Forecast data preparation
│   └── utils/
│       └── model_loader.py   # Model and data loading utilities
├── models/
│   ├── best_model.joblib          # Trained Random Forest model
│   ├── best_model_shap_explainer.joblib  # Cached SHAP explainer
│   ├── random_forest.joblib     # Random Forest model
│   ├── xgboost.joblib           # XGBoost model
│   ├── logistic_regression.joblib  # Logistic Regression model
│   └── preprocessor.joblib      # Fitted BRFSS preprocessor
├── data/
│   ├── raw/                  # Raw BRFSS and WHO datasets
│   └── processed/            # Processed forecasting data
├── outputs/
│   ├── reports/              # Pre-computed analysis reports
│   └── figures/              # Chart images and figures
└── app/
    ├── app.py              # Main application entry point
    └── requirements.txt    # Python dependencies
```

## Datasets and Sources

### Module 1: Individual Risk Prediction
- **Dataset**: CDC BRFSS 2015 (`diabetes_binary_health_indicators_BRFSS2015.csv`)
- **Records**: 229,474 (after removing 9.5% duplicates)
- **Target**: Diabetes_binary (0 = No diabetes, 1 = Diabetes)
- **Prevalence**: 15.3%
- **Features**: 21 (14 binary, 4 ordinal, 3 continuous)
  - Binary: HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex
  - Ordinal: GenHlth, Age, Education, Income
  - Continuous: BMI, MentHlth, PhysHlth
- **Split**: Stratified 80/20 train/test
- **Note**: Self-reported survey data; no glucose/HbA1c measurements

### Module 2: Population Analytics
- **Dataset**: Same BRFSS 2015 data, analysis of pre-computed reports
- **Outputs**: `outputs/reports/` — prevalence by demographic group, association measures (Cramér's V)

### Module 3: Prevalence Forecasting
- **Dataset 1**: WHO Global Health Observatory — India diabetes prevalence (`data/processed/india_forecasting_data.csv`)
  - 33 annual observations (1990–2022), age-standardized prevalence %
  - Uncertainty intervals included
- **Dataset 2**: ARIMA forecast artifacts (`outputs/reports/`)
  - ARIMA(2,2,0) selected model
  - 10-year forecast (2023–2032) with prediction intervals
  - Test period (2016–2022) held-out evaluation

## Models Used

### Module 1: Individual Risk Prediction
- **Primary model**: Random Forest (Test ROC-AUC: 0.817)
- **Other models**: Logistic Regression, Decision Tree, XGBoost
- **Best model**: Random Forest
- **Preprocessing**: BRFSSPreprocessor (one-hot encoding, scaling)
- **Prediction**: `predict_proba()` returns probability of diabetes (0–1)
- **Risk categories**: Low (<0.2), Moderate (0.2–0.5), High (0.5–0.8), Very High (≥0.8)

### Module 3: Prevalence Forecasting
- **Primary model**: ARIMA(2,2,0)
- **Other models**: Naive, Linear Trend, XGBoost
- **Best model**: ARIMA(2,2,0)
- **Validation**: 
  - Validation-period MAE (2008–2015): ARIMA had lowest error
  - Test-period MAE (2016–2022): 0.465 percentage points
  - Test RMSE: 0.586, Test MAPE: 2.17%

## Evaluation Metrics

### Module 1
- **ROC-AUC**: 0.817 (Random Forest on held-out test set)
- **Precision/Recall/F1**: Reported per class in model comparison reports
- **Risk category accuracy**: Verified against direct model calls

### Module 3
- **Test MAE**: 0.465 percentage points
- **Test RMSE**: 0.586 percentage points
- **Test MAPE**: 2.17%
- **Forecast MAE**: 0.465 (validation), 0.465 (test)

## SHAP Explainability

### What SHAP Tells Us
- SHAP (SHapley Additive exPlanations) values quantify each feature's contribution to the model's prediction
- Values are in **log-odds** units and are **additive**: base value + sum of contributions = model's raw output for the diabetes class
- **Positive contribution (red)**: feature pushes prediction toward diabetes class
- **Negative contribution (blue)**: feature pushes prediction away from diabetes class

### Pages Powered by SHAP
- **Explainable AI page**: Global feature importance (mean |SHAP| per feature), SHAP beeswarm plot (per-sample contributions), and per-instance explanation (waterfall chart)
- **Key insight**: The SHAP explainer is cached (`models/best_model_shap_explainer.joblib`) so the expensive background computation runs only once

### SHAP Pages
- **Global feature importance**: Bar chart of average |SHAP| values across background samples
- **SHAM summary**: Beeswarm plot showing per-sample contributions, colored by feature value
- **Per-instance explanation**: Waterfall chart for a specific patient prediction, showing which features push risk up or down

## What-If Simulation

### What-If Simulation Page
- **Purpose**: Explore how changing 21 BRFSS features moves the model's predicted risk
- **Interface**: Two columns — Baseline (reference patient) and Scenario (editable scenario)
- **Results shown**:
  - Baseline probability + risk category
  - Scenario probability + risk category
  - Difference in percentage points (with direction arrow)
  - Feature-level comparison table (baseline vs scenario values)
- **Disclaimer**: *This is a model simulation — it does not predict a medical outcome nor establish causation. The change in probability reflects how the trained model re-weights the 21 BRFSS features. Consult a healthcare provider for medical decisions.*

### How to Use
1. Fill in the **Baseline** patient profile (default values shown)
2. Edit the **Scenario** inputs (e.g., lower BMI, quit smoking, change health rating)
3. Click **Run simulation** to see how the predicted probability changes
4. Review the feature comparison table and delta explanation

## Streamlit Run Instructions

```bash
cd C:\Users\ALWIN ABHISHEK\DiabPredict-AI
streamlit run app\app.py
```

The application will be available at `http://localhost:8501`.

### Navigation
Use the sidebar radio to switch between pages:
- 🏠 Home / Overview
- 🎯 Individual Risk Prediction
- 📊 Population Analytics
- 🔮 Prevalence Forecast
- 🔍 Explainable AI
- 🧪 What-If Simulation

## Limitations and Medical Disclaimer

### Important Limitations

| Area | Limitation |
|------|------------|
| **Module 1** | BRFSS is cross-sectional survey data; self-reported; no glucose/HbA1c; no temporal component |
| **Module 2** | Association ≠ causation; no survey weights applied; complete-case analysis only |
| **Module 3** | WHO estimates are modeled (not direct surveys); I(2) trend may over-project; no covariates |
| **All** | Models are screening/forecasting tools — **NOT diagnostic or clinical decision systems** |

### Medical Disclaimer

⚠️ **This application provides risk predictions and forecasts based on machine learning models. It does NOT diagnose diabetes or replace professional medical advice. All predictions are model estimates and should be interpreted as screening support only. Consult a healthcare provider for medical decisions.**

The models are trained on self-reported CDC BRFSS 2015 survey data and WHO modeled estimates. They are designed for screening and research purposes only and should not be used for clinical decision-making.

## Streamlit Commands

```bash
# Run the application
streamlit run app\app.py

# Check available commands
streamlit --help
```

## Citation

If you use this code or data in your research, please cite:

- BRFSS 2015 Dataset: Centers for Disease Control and Prevention (CDC)
- WHO Global Health Observatory: World Health Organization
- ARIMA forecasting: Box, Jenkins, and Reinsel (2016)

---

*DiabPredict AI v0.1.0 — Built with Streamlit, scikit-learn, XGBoost, statsmodels, SHAP, and Plotly.*