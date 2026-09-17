"""
Streamlit dashboard for DiabPredict AI.

Diabetes Risk Prediction & Population Prevalence Forecasting System
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Page config
st.set_page_config(
    page_title="DiabPredict AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.3rem;
        font-weight: 500;
        color: #333;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #664d03;  /* Dark text for readability on yellow background */
    }
    .disclaimer strong {
        color: #664d03;
    }
    .module-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1.5rem;
        height: 100%;
    }
    .module-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-ready { background-color: #d4edda; color: #155724; }
    .status-pending { background-color: #fff3cd; color: #856404; }
    .result-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .risk-low { background-color: #d4edda; border-color: #c3e6cb; color: #155724; }
    .risk-moderate { background-color: #fff3cd; border-color: #ffeeba; color: #856404; }
    .risk-high { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; }
    .risk-very-high { background-color: #f5c6cb; border-color: #f1b0b7; color: #721c24; }
</style>
""", unsafe_allow_html=True)


def show_disclaimer():
    """Show medical disclaimer."""
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> This application provides risk predictions and forecasts 
        based on machine learning models. It does <strong>NOT</strong> diagnose diabetes or replace 
        professional medical advice. All predictions are model estimates and should be interpreted 
        as screening support only. Consult a healthcare provider for medical decisions.
    </div>
    """, unsafe_allow_html=True)


def check_artifacts():
    """Check if model and data artifacts exist."""
    from pathlib import Path
    base = Path(__file__).parent.parent
    
    checks = {
        "Best Model": base / "models" / "best_model.joblib",
        "Preprocessor": base / "models" / "preprocessor.joblib",
        "BRFSS Data (Module 1)": base / "data" / "raw" / "diabetes_binary_health_indicators_BRFSS2015.csv",
        "WHO Data (Module 3)": base / "data" / "raw" / "who_diabetes_prevalence_agestd.csv",
        "Population Reports": base / "outputs" / "reports" / "prevalence_summary_all_features.csv",
        "Forecast Results": base / "outputs" / "reports" / "arima_future_forecast.csv",
    }
    
    results = {}
    for name, path in checks.items():
        results[name] = path.exists()
    return results


def load_predictor_cached():
    """Load the predictor with caching."""
    from src.utils.model_loader import load_predictor
    return load_predictor()


def individual_risk_page():
    """Individual Diabetes Risk Prediction page."""
    st.markdown('<div class="main-header">🎯 Individual Diabetes Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Screening Support Tool — Not a Medical Diagnosis</div>', unsafe_allow_html=True)
    
    show_disclaimer()
    
    # Load feature metadata
    from src.predict import BRFSS_FEATURES, FEATURE_LABELS, FEATURE_OPTIONS, FEATURE_RANGES
    
    # Load predictor
    try:
        predictor = load_predictor_cached()
        st.success("✅ Model and preprocessor loaded successfully")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()
    
    # Input form
    with st.form("risk_prediction_form"):
        st.subheader("Enter Patient Information")
        
        # Section 1: Health Indicators
        st.markdown("### 🩺 Health Indicators")
        col1, col2 = st.columns(2)
        
        with col1:
            high_bp = st.selectbox(
                FEATURE_LABELS['HighBP'],
                options=[0, 1],
                format_func=lambda x: FEATURE_OPTIONS['HighBP'][x],
                index=0
            )
            high_chol = st.selectbox(
                FEATURE_LABELS['HighChol'],
                options=[0, 1],
                format_func=lambda x: FEATURE_OPTIONS['HighChol'][x],
                index=0
            )
            chol_check = st.selectbox(
                FEATURE_LABELS['CholCheck'],
                options=[0, 1],
                format_func=lambda x: FEATURE_OPTIONS['CholCheck'][x],
                index=1
            )
            stroke = st.selectbox(
                FEATURE_LABELS['Stroke'],
                options=[0, 1],
                format_func=lambda x: FEATURE_OPTIONS['Stroke'][x],
                index=0
            )
            heart_disease = st.selectbox(
                FEATURE_LABELS['HeartDiseaseorAttack'],
                options=[0, 1],
                format_func=lambda x: FEATURE_OPTIONS['HeartDiseaseorAttack'][x],
                index=0
            )
        
        with col2:
            diff_walk = st.selectbox(
                FEATURE_LABELS['DiffWalk'],
                options=[0, 1],
                format_func=lambda x: FEATURE_OPTIONS['DiffWalk'][x],
                index=0
            )
            gen_hlth = st.selectbox(
                FEATURE_LABELS['GenHlth'],
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: FEATURE_OPTIONS['GenHlth'][x],
                index=2
            )
            ment_hlth = st.number_input(
                FEATURE_LABELS['MentHlth'],
                min_value=FEATURE_RANGES['MentHlth']['min'],
                max_value=FEATURE_RANGES['MentHlth']['max'],
                value=FEATURE_RANGES['MentHlth']['mean'],
                step=1
            )
            phys_hlth = st.number_input(
                FEATURE_LABELS['PhysHlth'],
                min_value=FEATURE_RANGES['PhysHlth']['min'],
                max_value=FEATURE_RANGES['PhysHlth']['max'],
                value=FEATURE_RANGES['PhysHlth']['mean'],
                step=1
            )
        
        st.divider()
        
        # Section 2: Lifestyle
        st.markdown("### 🏃 Lifestyle Factors")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            phys_activity = st.selectbox(
                FEATURE_LABELS['PhysActivity'],
                options=[0, 1],
                format_func=lambda x: FEATURE_OPTIONS['PhysActivity'][x],
                index=1
            )
            fruits = st.selectbox(
                FEATURE_LABELS['Fruits'],
                options=[0, 1],
                format_func=lambda x: FEATURE_OPTIONS['Fruits'][x],
                index=1
            )
        
        with col2:
            veggies = st.selectbox(
                FEATURE_LABELS['Veggies'],
                options=[0, 1],
                format_func=lambda x: FEATURE_OPTIONS['Veggies'][x],
                index=1
            )
            smoker = st.selectbox(
                FEATURE_LABELS['Smoker'],
                options=[0, 1],
                format_func=lambda x: FEATURE_OPTIONS['Smoker'][x],
                index=0
            )
        
        with col3:
            hvy_alcohol = st.selectbox(
                FEATURE_LABELS['HvyAlcoholConsump'],
                options=[0, 1],
                format_func=lambda x: FEATURE_OPTIONS['HvyAlcoholConsump'][x],
                index=0
            )
        
        st.divider()
        
        # Section 3: Demographics
        st.markdown("### 👤 Demographics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sex = st.selectbox(
                FEATURE_LABELS['Sex'],
                options=[0, 1],
                format_func=lambda x: FEATURE_OPTIONS['Sex'][x],
                index=1
            )
            age = st.selectbox(
                FEATURE_LABELS['Age'],
                options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                format_func=lambda x: FEATURE_OPTIONS['Age'][x],
                index=6
            )
        
        with col2:
            education = st.selectbox(
                FEATURE_LABELS['Education'],
                options=[1, 2, 3, 4, 5, 6],
                format_func=lambda x: FEATURE_OPTIONS['Education'][x],
                index=3
            )
            income = st.selectbox(
                FEATURE_LABELS['Income'],
                options=[1, 2, 3, 4, 5, 6, 7, 8],
                format_func=lambda x: FEATURE_OPTIONS['Income'][x],
                index=4
            )
        
        with col3:
            bmi = st.number_input(
                FEATURE_LABELS['BMI'],
                min_value=FEATURE_RANGES['BMI']['min'],
                max_value=FEATURE_RANGES['BMI']['max'],
                value=FEATURE_RANGES['BMI']['mean'],
                step=0.1,
                format="%.1f"
            )
        
        st.divider()
        
        # Section 4: Healthcare Access
        st.markdown("### 🏥 Healthcare Access")
        col1, col2 = st.columns(2)
        
        with col1:
            any_healthcare = st.selectbox(
                FEATURE_LABELS['AnyHealthcare'],
                options=[0, 1],
                format_func=lambda x: FEATURE_OPTIONS['AnyHealthcare'][x],
                index=1
            )
        
        with col2:
            no_doc_cost = st.selectbox(
                FEATURE_LABELS['NoDocbcCost'],
                options=[0, 1],
                format_func=lambda x: FEATURE_OPTIONS['NoDocbcCost'][x],
                index=0
            )
        
        # Predict button
        submitted = st.form_submit_button("🔍 Predict Risk", type="primary", use_container_width=True)
    
    # Process prediction
    if submitted:
        # Collect all inputs in correct order
        input_data = {
            'HighBP': high_bp,
            'HighChol': high_chol,
            'CholCheck': chol_check,
            'Smoker': smoker,
            'Stroke': stroke,
            'HeartDiseaseorAttack': heart_disease,
            'PhysActivity': phys_activity,
            'Fruits': fruits,
            'Veggies': veggies,
            'HvyAlcoholConsump': hvy_alcohol,
            'AnyHealthcare': any_healthcare,
            'NoDocbcCost': no_doc_cost,
            'DiffWalk': diff_walk,
            'Sex': sex,
            'GenHlth': gen_hlth,
            'Age': age,
            'Education': education,
            'Income': income,
            'BMI': bmi,
            'MentHlth': ment_hlth,
            'PhysHlth': phys_hlth
        }
        
        # Validate inputs
        missing = [k for k, v in input_data.items() if v is None]
        if missing:
            st.error(f"Missing required inputs: {missing}")
        else:
            try:
                with st.spinner("Computing prediction..."):
                    probability = predictor.predict_proba(input_data)
                    pred_class = predictor.predict(input_data)
                    risk_category = predictor.get_risk_category(probability)
                
                # Display results
                st.divider()
                st.subheader("📋 Prediction Results")
                
                # Risk category card
                risk_class_map = {
                    "Low Risk": "risk-low",
                    "Moderate Risk": "risk-moderate",
                    "High Risk": "risk-high",
                    "Very High Risk": "risk-very-high"
                }
                risk_class = risk_class_map.get(risk_category, "")
                
                st.markdown(f"""
                <div class="result-card {risk_class}">
                    <h3 style="margin: 0;">{risk_category}</h3>
                    <p style="margin: 0.5rem 0; font-size: 1.1rem;">
                        Predicted Probability: <strong>{probability:.1%}</strong> | 
                        Predicted Class: <strong>{"Diabetes" if pred_class == 1 else "No Diabetes"}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Diabetes Risk Probability", f"{probability:.1%}")
                with col2:
                    st.metric("Predicted Class", "Diabetes" if pred_class == 1 else "No Diabetes")
                with col3:
                    st.metric("Risk Category", risk_category)
                
                # Disclaimer reminder
                st.warning("""
                ⚠️ **Reminder:** This is a machine-learning screening/risk model trained on CDC BRFSS 2015 survey data.
                It is **NOT a medical diagnosis**. The prediction is a statistical risk estimate based on self-reported
                survey variables. Consult a healthcare provider for medical evaluation and decisions.
                """)
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")


def home_page():
    """Home/Overview page."""
    st.markdown('<div class="main-header">DiabPredict AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Diabetes Risk Prediction & Population Prevalence Forecasting System</div>', unsafe_allow_html=True)
    
    show_disclaimer()
    
    # Artifact status
    st.subheader("System Status")
    artifacts = check_artifacts()
    
    cols = st.columns(3)
    for i, (name, exists) in enumerate(artifacts.items()):
        with cols[i % 3]:
            status_class = "status-ready" if exists else "status-pending"
            status_text = "Ready" if exists else "Missing"
            st.markdown(f"""
            <div class="module-card">
                <div class="module-title">{name}</div>
                <span class="status-badge {status_class}">{status_text}</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Three modules
    st.subheader("Three Core Modules")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="module-card">
            <div class="module-title">🎯 Module 1: Individual Risk Prediction</div>
            <ul>
                <li>Binary classification (diabetes yes/no)</li>
                <li>CDC BRFSS 2015 dataset (229K records)</li>
                <li>21 features: demographics, health indicators, lifestyle</li>
                <li>Models: Logistic Regression, Decision Tree, Random Forest, XGBoost</li>
                <li>Best: Random Forest (Test ROC-AUC: 0.817)</li>
                <li>SHAP explainability & What-if simulation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="module-card">
            <div class="module-title">📊 Module 2: Population Trends</div>
            <ul>
                <li>Prevalence analysis by demographic groups</li>
                <li>Age, Sex, BMI categories, Education, Income</li>
                <li>Health indicators: HighBP, HighChol, Physical Activity</li>
                <li>General health, Smoking, Stroke, Heart disease</li>
                <li>Association analysis (Cramér's V)</li>
                <li>Interactive visualizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="module-card">
            <div class="module-title">🔮 Module 3: Prevalence Forecasting</div>
            <ul>
                <li>WHO GHO age-standardized prevalence (India, 1990–2022)</li>
                <li>33 annual observations, continuous</li>
                <li>Chronological train/validation/test split</li>
                <li>Models: Naive, Linear Trend, ARIMA, XGBoost</li>
                <li>Best: ARIMA(2,2,0) — Test MAE: 0.465</li>
                <li>10-year forecast (2023–2032) with prediction intervals</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset information
    st.subheader("Data Sources")
    
    tab1, tab2 = st.tabs(["Module 1: CDC BRFSS 2015", "Module 3: WHO GHO"])
    
    with tab1:
        st.markdown("""
        **CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015**
        - **Dataset**: `diabetes_binary_health_indicators_BRFSS2015.csv`
        - **Records**: 229,474 (after removing 9.5% duplicates)
        - **Target**: Diabetes_binary (0 = No diabetes, 1 = Diabetes)
        - **Prevalence**: 15.3%
        - **Features**: 21 (14 binary, 4 ordinal, 3 continuous)
        - **Split**: Stratified 80/20 train/test
        - **Note**: Self-reported survey data; no glucose/HbA1c measurements
        """)
    
    with tab2:
        st.markdown("""
        **WHO Global Health Observatory — Indicator: NCD_DIABETES_PREVALENCE_AGESTD**
        - **Dataset**: `who_diabetes_prevalence_agestd.csv`
        - **Country**: India (ISO: IND)
        - **Population**: Both sexes, Age 18+
        - **Years**: 1990–2022 (33 consecutive observations)
        - **Values**: Age-standardized prevalence % with 95% uncertainty intervals
        - **Split**: Train 1990–2007 (18), Val 2008–2015 (8), Test 2016–2022 (7)
        - **Trend**: 11.6% (1990) → 22.6% (2022)
        - **Note**: Modeled estimates, not direct survey measurements
        """)
    
    st.markdown("---")
    
    # Limitations
    st.subheader("Important Limitations")
    
    st.markdown("""
    | Area | Limitation |
    |------|------------|
    | **Module 1** | BRFSS is cross-sectional survey data; self-reported; no glucose/HbA1c; no temporal component |
    | **Module 2** | Association ≠ causation; no survey weights applied; complete-case analysis only |
    | **Module 3** | WHO estimates are modeled (not direct surveys); I(2) trend may over-project; no covariates |
    | **All** | Models are screening/forecasting tools — **NOT diagnostic or clinical decision systems** |
    """)
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate between modules")


def main():
    """Main application entry point."""
    # Sidebar
    with st.sidebar:
        st.title("🏥 DiabPredict AI")
        st.caption("Diabetes Risk Prediction & Population Prevalence Forecasting")
        st.divider()
        
        # Navigation
        page = st.radio(
            "Navigate",
            [
                "🏠 Home / Overview",
                "🎯 Individual Risk Prediction",
                "📊 Population Analytics",
                "🔮 Prevalence Forecast",
                "🔍 Explainable AI",
                "🧪 What-If Simulation"
            ],
            index=0
        )
        
        st.divider()
        
        # Module status
        st.subheader("Module Status")
        artifacts = check_artifacts()
        
        for name, exists in artifacts.items():
            icon = "✅" if exists else "⏳"
            st.caption(f"{icon} {name}")
        
        st.divider()
        
        st.info(
            "DiabPredict AI v0.1.0\n\n"
            "End-to-end ML system for diabetes risk prediction "
            "and population prevalence forecasting.\n\n"
            "Built with: Streamlit, scikit-learn, XGBoost, "
            "statsmodels, SHAP, Plotly"
        )
    
    # Route to appropriate page
    if page == "🏠 Home / Overview":
        home_page()
    elif page == "🎯 Individual Risk Prediction":
        individual_risk_page()
    elif page == "📊 Population Analytics":
        from src.population_analytics import population_analytics_page
        population_analytics_page()
    elif page == "🔮 Prevalence Forecast":
        from src.prevalence_forecast import prevalence_forecast_page
        prevalence_forecast_page()
    elif page == "🔍 Explainable AI":
        from src.explainability_page import explainability_page
        explainability_page()
    elif page == "🧪 What-If Simulation":
        from src.what_if_simulation import what_if_simulation_page
        what_if_simulation_page()


if __name__ == "__main__":
    main()
