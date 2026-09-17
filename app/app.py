"""
Streamlit dashboard for DiabPredict AI.

Diabetes Risk Prediction & Population Prevalence Forecasting System
"""

import streamlit as st
import sys
from pathlib import Path

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