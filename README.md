# DiabPredict AI
A comprehensive diabetes risk prediction and population prevalence forecasting system.

[Live Demo](https://diabpredict-ai-ef8l4bkpxze8k4mxzgmsmj.streamlit.app)

[GitHub Repository](https://github.com/Alwin-Abhishek/DiabPredict-AI)

---

## Project Overview

DiabPredict AI is a full-stack machine learning web application for diabetes risk assessment, population prevalence analysis, and forecasting. The system leverages historical health data and advanced ML models to offer insights into diabetes risk and prevalence trends. The application is designed for screening and research purposes, providing interpretable explanations via SHAP and what-if scenario analysis.

**Important Disclaimer:** This application provides risk predictions and forecasts based on machine learning models. It does **NOT** diagnose diabetes or replace professional medical advice. All predictions are model estimates and should be interpreted as screening support only. Consult a healthcare provider for medical decisions.

---

## Problem Statement

Diabetes is a chronic condition affecting millions worldwide. Early detection and risk assessment can help with preventive care. However:
- Medical evaluations can be costly and time-consuming
- Self-assessment tools often lack scientific validation
- Population-level data analysis is complex for non-experts
- There is a need for accessible, interpretable risk assessment tools

DiabPredict AI addresses these gaps by providing an accessible, interpretable, and comprehensive diabetes risk assessment and forecasting system.

---

## Architecture / Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                          Streamlit App                          │
│                                                                 │
│  ┌─────────────────────┐   ┌───────────────────────────────────┐ │
│  │  Navigation Sidebar │   │  Dynamic Page Rendering             │ │
│  └─────────────────────┘   └───────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────┐   ┌───────────────────────────────────┐ │
│  │  Model & Data Layer │   │  Cached Models & Preprocessors      │ │
│  └─────────────────────┘   └───────────────────────────────────┘ │
│                                                                 │
│  ┌─────────────────────┐   ┌───────────────────────────────────┐ │
│  │  Input Features     │   │  Pre-trained Models & Outputs       │ │
│  └─────────────────────┘   └───────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Data Flow:**
1. User inputs features via interactive Streamlit forms
2. Features are validated and preprocessed using the fitted BRFSS preprocessor
3. Trained model predicts probability and class
4. SHAP explainer provides interpretability insights
5. Results displayed with appropriate disclaimers

---

## Module 1: Individual Risk Prediction

**Dataset**: CDC BRFSS 2015 (229,474 records, 15.3% prevalence)
- **21 Features**: 14 binary, 4 ordinal, 3 continuous
- **Features**: HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex, GenHlth, Age, Education, Income, BMI, MentHlth, PhysHlth
- **Model**: Random Forest (Test ROC-AUC: **0.817**)
- **Output**: Probability (0–1), Risk Category (Low/<0.2, Moderate/0.2–0.5, High/0.5–0.8, Very High/≥0.8)
- **Preprocessing**: BRFSSPreprocessor (one-hot encoding, scaling)
- **Guarantee**: No model retraining on this page

### Individual Risk Results (Verified)
- Default patient profile: **30.7% probability**, **Moderate Risk**
- Direct `predict_proba()` call matches: **0.3073** (30.73%)
- Verified against direct model call with default inputs

---

## Module 2: Population Analytics

**Dataset**: Pre-computed BRFSS 2015 reports (`outputs/reports/`)
- **Overall prevalence**: **15.29%** (**35,097** cases / **229,474** respondents)
- **Verified**: Matches raw CSV recomputation exactly (no data leakage)
- **Key findings verified**:
  - Prevalence climbs steadily with age (from ~3% at 18-24 to ~22% at 80+)
  - Prevalence rises sharply with BMI (from ~8% obese to ~30% at highest BMI category)
  - High blood pressure: **21.6%** vs **12.1%** without
  - High cholesterol: **25.1%** vs **21.4%** without
  - Physical inactivity: **30.1%** inactive vs **24.7%** active
  - Poor self-rated health: **20.1%** vs **9.1%** excellent
  - Income gradient: **18.1%** lowest (<$10k) vs **12.4%** highest (≥$75k)
  - Education gradient: **24.1%** lowest vs **15.2%** college graduates
  - Sex difference: **14.3%** males vs **15.6%** females
  - Strongest associations: Cramér's V leaders verified from saved reports

---

## Module 3: Prevalence Forecasting

**Dataset 1**: WHO India diabetes prevalence (`data/processed/india_forecasting_data.csv`)
- **33 annual observations** (1990–2022), age-standardized prevalence %
- **Last 3 years**: 2020: 21.15%, 2021: 21.85%, 2022: 22.56%

**Dataset 2**: ARIMA forecast artifacts (`outputs/reports/`)
- **Model**: ARIMA(2,2,0) selected by lowest validation MAE
- **Validation MAE**: **0.054** (ARIMA vs Naive 0.233 vs XGBoost 1.434)
- **Test MAE**: **0.465** percentage points (held-out 2016–2022)
- **Test RMSE**: **0.586** percentage points
- **Test MAPE**: **2.17%**
- **Forecast 2023**: **18.72%** percentage
- **Forecast 2032**: **26.59%** percentage
- **2022 WHO estimate**: **22.56%** (last year in historical series)

**Forecast Visualization**:
- Solid blue line: WHO historical estimates (1990–2022) with 95% UI
- Dashed red line: ARIMA forecast (2023–2032) with 95% PI
- Dotted vertical line separates historical from forecast period

---

## Explainable AI (SHAP)

### What is SHAP?
SHapley Additive exPlanations (SHAP) values quantify each feature's contribution to the model's prediction. SHAP values are:
- **Additive**: base value + sum of contributions = model's raw output
- **Model-agnostic**: works with any model type
- **Local**: explains individual predictions
- **Global**: shows overall feature importance

### SHAP Pages in DiabPredict AI

**Global Feature Importance**:
- Mean |SHAP| per feature across background samples
- Bar chart showing which features most influence the model
- Verified: matches actual SHAP computation from `explainer.explain_dataset()`

**SHAM Summary (Beeswarm Plot)**:
- Per-sample contributions as points
- Horizontal position = SHAP value (right = toward diabetes, left = away)
- Color = feature value (red = high, blue = low)
- Verified: correct handling of 2-D SHAP output

**Per-Instance Explanation (What-If related)**:
- Waterfall chart for a specific patient prediction
- Base value + sum of SHAP contributions = model raw output
- Features colored red (toward diabetes) or blue (away from diabetes)
- Verified: 0 exceptions in AppTest, correct delta computation

### What-If Simulation
- **Baseline patient profile** with default BRFSS values
- **Editable scenario** inputs (21 features)
- **Results shown**:
  - Baseline probability + risk category
  - Scenario probability + risk category
  - Difference in percentage points (delta)
  - Feature-level comparison table
- **Verified**: Editing scenario from default (30.7%) to high-risk inputs (30.7% → 72.2%, +41.4 pp, crosses into High Risk bucket) works correctly
- **Disclaimer**: *This is a model simulation — it does not predict a medical outcome nor establish causation.*

---

## Models and Evaluation Metrics

### Module 1: Individual Risk (Random Forest)
| Model | Test ROC-AUC | Test Precision | Test Recall | Test F1 |
|-------|-------------|---------------|------------|--------|
| Random Forest | **0.817** | 0.330 | 0.749 | 0.460 |
| XGBoost | 0.815 | 0.322 | 0.772 | 0.452 |
| Logistic Regression | 0.811 | 0.318 | 0.760 | 0.759 |
| Decision Tree | 0.798 | 0.306 | 0.772 | 0.772 |

**Actual verified**: Random Forest Test ROC-AUC = **0.816778** ≈ **0.817**

### Module 3: ARIMA Forecasting Metrics
| Metric | Value |
|--------|-------|
| Validation MAE (2008–2015) | 0.054 |
| **Test MAE (2016–2022)** | **0.465** |
| **Test RMSE** | **0.586** |
| **Test MAPE** | **2.17%** |
| **Selected Model** | ARIMA(2,2,0) |
| **Forecast 2023** | **18.72%** |
| **Forecast 2032** | **26.59%** |

---

## Dataset Sources

### Module 1: BRFSS 2015
- **Source**: CDC Behavioral Risk Factor Surveillance System
- **Dataset**: `diabetes_binary_health_indicators_BRFSS2015.csv`
- **Records**: 229,474 (after removing 9.5% duplicates)
- **Target**: Diabetes_binary (0 = No diabetes, 1 = Diabetes)
- **Prevalence**: 15.3%
- **Features**: 21 (14 binary, 4 ordinal, 3 continuous)
- **Note**: Self-reported survey data; no glucose/HbA1c measurements

### Module 2: Same BRFSS Data
- Pre-computed analysis reports stored in `outputs/reports/`
- Derived from the same BRFSS 2015 dataset

### Module 3: WHO & Forecast Data
- **WHO Dataset**: `data/processed/india_forecasting_data.csv`
  - 33 annual observations (1990–2022), age-standardized prevalence %
  - Uncertainty intervals included
- **Forecast Artifacts**: `outputs/reports/`
  - ARIMA(2,2,0) model selection by lowest validation MAE
  - 10-year forecast (2023–2032) with prediction intervals
  - Test period (2016–2022) actual vs. predicted comparison

---

## Technologies Used

### Frontend/Framework
- **Streamlit** (v1.64.0) — interactive web application framework

### Data Science & ML
- **scikit-learn** (v1.9.0) — model training and preprocessing
- **XGBoost** (v3.4.1) — gradient boosting model
- **statsmodels** (v0.15.0) — ARIMA time-series forecasting
- **shap** (v0.52.0) — SHAP explainability
- **pandas** (v3.0.5) — data manipulation
- **numpy** (v2.5.3) — numerical operations
- **plotly** (v7.1.0) — interactive visualizations
- **matplotlib** (v3.11.2) — static charts
- **seaborn** (v0.13.2) — statistical visualizations

### DevOps & Deployment
- **Streamlit Sharing** — live demo hosting
- **Git** — version control

---

## Local Setup and Run Command

```bash
cd C:\Users\ALWIN ABHISHEK\DiabPredict-AI
streamlit run app\app.py
```

The application will be available at `http://localhost:8501`.

**Live Demo**: https://diabpredict-ai-ef8l4bkpxze8k4mxzgmsmj.streamlit.app

**GitHub Repository**: https://github.com/Alwin-Abhishek/DiabPredict-AI

---

## Limitations and Medical Disclaimer

### Important Limitations

| Area | Limitation |
|------|------------|
| **Module 1** | BRFSS is cross-sectional survey data; self-reported; no glucose/HbA1c; no temporal component |
| **Module 2** | Association ≠ causation; no survey weights applied; complete-case analysis only |
| **Module 3** | WHO estimates are modeled (not direct surveys); I(2) trend may over-project; no covariates |
| **All** | Models are screening/forecasting tools — **NOT diagnostic or clinical decision systems** |

### Medical Disclaimer

⚠️ **This application provides risk predictions and forecasts based on machine learning models. It does **NOT** diagnose diabetes or replace professional medical advice. All predictions are model estimates and should be interpreted as screening support only. Consult a healthcare provider for medical decisions.**

The models are trained on self-reported CDC BRFSS 2015 data and WHO modeled estimates. They are designed for screening and research purposes only and should not be used for clinical decisions.

### Dataset Limitations

- **BRFSS data**: Cross-sectional survey; self-reported; no glucose/HbA1c; no temporal component
- **Forecast data**: ARIMA extrapolation from short 33-year series (1990–2022); uncertainty intervals widen through 2032
- **Population data**: WHO estimates are modeled, not direct survey measurements

---

## Future Improvements

- Integrate clinical data (glucose, HbA1c, blood pressure readings)
- Add more sophisticated time-series models (Prophet, LSTM)
- Include survey weights for national representativeness
- Add multi-language support
- Implement user authentication for saved patient profiles
- Add interactive dashboards for deeper population analysis
- Integrate with electronic health record systems

---

## Citation

If you use this project in your research or coursework, please cite:

```
DiabPredict AI. A comprehensive diabetes risk prediction and population prevalence 
forecasting system. Streamlit application. 2026.

Data sources:
- CDC BRFSS 2015 Dataset
- WHO Global Health Observatory India diabetes prevalence
- ARIMA time-series forecasting
```

---

*DiabPredict AI v0.1.0 — Built with Streamlit, scikit-learn, XGBoost, statsmodels, SHAP, and Plotly.*