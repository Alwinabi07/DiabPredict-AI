"""Explainable AI page — SHAP explanations for the trained Module 1 model.

Uses the existing ``models/best_model.joblib`` (Random Forest) and
``models/preprocessor.joblib`` together with the SHAP utilities in
``src/explainability.py``.

The model is never retrained here. The fitted SHAP explainer is cached for
the session and persisted to ``models/best_model_shap_explainer.joblib`` so
the expensive background computation is only done once.

SHAP values are reported in the model's raw output space (log-odds / margin):
positive contributions push the prediction toward the diabetes class, negative
contributions push it away.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.explainability import ModelExplainer, create_explainer
from src.predict import (
    BRFSS_FEATURES,
    FEATURE_LABELS,
    FEATURE_OPTIONS,
    FEATURE_RANGES,
)
from src.utils.model_loader import (
    get_training_sample,
    load_best_model,
    load_preprocessor,
    load_predictor,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
EXPLAINER_PATH = MODELS_DIR / "best_model_shap_explainer.joblib"
GLOBAL_BACKGROUND_SAMPLES = 200

POSITIVE_COLOR = "#d62728"
NEGATIVE_COLOR = "#1f77b4"


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading trained model and preprocessor...")
def _load_predictor_cached():
    return load_predictor()


@st.cache_resource(show_spinner="Fitting SHAP explainer (first load only)...")
def _load_global_explanations() -> Dict[str, Any]:
    """Load the fitted explainer (or create and save it) and global SHAP values."""
    np.random.seed(42)

    model = load_best_model()
    preprocessor = load_preprocessor()
    feature_names = preprocessor.get_feature_names_out()
    X_background = get_training_sample(
        n_samples=GLOBAL_BACKGROUND_SAMPLES, random_state=42
    )

    explainer: Optional[ModelExplainer] = None
    if EXPLAINER_PATH.exists():
        try:
            explainer = joblib.load(EXPLAINER_PATH)
        except Exception:
            explainer = None

    if explainer is None:
        explainer = create_explainer(model, feature_names, X_background)
        os.makedirs(EXPLAINER_PATH.parent, exist_ok=True)
        joblib.dump(explainer, EXPLAINER_PATH)

    shap_values = explainer.explain_dataset(X_background)

    # Normalise shape so downstream 1-D operations (DataFrame columns,
    # plotly axes) always work regardless of how SHAP reports output:
    #   - legacy SHAP lists -> pick the diabetes (positive) class
    #   - (samples, features, classes) -> keep the diabetes class
    #   - (samples, features) -> as-is
    if isinstance(shap_values, list):
        shap_values = (
            shap_values[1] if len(shap_values) > 1 else shap_values[0]
        )
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3 and shap_values.shape[-1] > 1:
        shap_values = shap_values[..., 1]
    elif shap_values.ndim == 3:
        shap_values = shap_values[..., 0]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_abs_shap = np.asarray(mean_abs_shap).ravel()

    return {
        "explainer": explainer,
        "model": model,
        "feature_names": feature_names,
        "X_background": X_background,
        "shap_values": shap_values,
        "mean_abs_shap": mean_abs_shap,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _normalize_mean_abs_shap(shap_like: Any, feature_names: list) -> np.ndarray:
    """Reduce any SHAP output shape to one mean-|SHAP| value per feature.

    Handles how different SHAP/Explainer versions and class-aware tree models
    report output:
      - list of arrays  -> use the diabetes (positive) class element
      - 1-D arrays      -> already one value per feature (magnitudes applied)
      - 2-D arrays      -> (samples, features) -> mean |value| over samples;
                           (features, classes) -> pick the diabetes class
      - 3-D arrays      -> (samples, features, classes) -> pick the diabetes
                           class then mean |value| over samples
    Always returns a 1-D array whose length is at most ``len(feature_names)``.
    """
    if isinstance(shap_like, list):
        shap_like = (
            shap_like[1] if len(shap_like) > 1 else shap_like[0]
        )
    values = np.asarray(shap_like, dtype=float)

    if values.ndim >= 3:
        values = values[..., 1] if values.shape[-1] > 1 else values[..., 0]
        values = np.abs(values).mean(axis=0)
    elif values.ndim == 2:
        # Two candidate layouts: columns are samples or classes.
        if values.shape[1] == 2 and values.shape[0] >= 2:
            values = np.abs(values[:, 1])
        else:
            values = np.abs(values).mean(axis=0)
    elif values.ndim == 1:
        values = np.abs(values)
    else:
        return np.zeros(len(feature_names))

    flat = np.asarray(values).ravel()[: len(feature_names)]
    if len(flat) < len(feature_names):
        flat = np.concatenate(
            [flat, np.zeros(len(feature_names) - len(flat))]
        )
    return flat


def _base_value(explainer: ModelExplainer) -> float:
    """Base (expected) model output for the diabetes class, in log-odds."""
    expected = np.atleast_1d(explainer.explainer.expected_value)
    return float(expected[1]) if len(expected) > 1 else float(expected[0])


def _format_feature_value(feature: str, raw_value: Any) -> str:
    """Human-readable display of a user-entered feature value."""
    options = FEATURE_OPTIONS.get(feature)
    if options:
        try:
            return str(options[int(raw_value)])
        except (KeyError, ValueError):
            return str(raw_value)
    return str(raw_value)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def build_importance_figure(
    mean_abs_shap: np.ndarray, feature_names: list
) -> go.Figure:
    """Global feature importance: mean absolute SHAP value per feature."""
    # SHAP may report output as a 1-D, 2-D, or 3-D array (or a list of
    # arrays). Normalise to exactly one mean-|SHAP| value per feature so the
    # per-feature DataFrame and plotly axes stay 1-D.
    mean_abs_shap = _normalize_mean_abs_shap(mean_abs_shap, feature_names)
    df = pd.DataFrame(
        {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
    ).sort_values("mean_abs_shap", ascending=True)

    fig = px.bar(
        df,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        color="mean_abs_shap",
        color_continuous_scale="Blues",
        text=df["mean_abs_shap"].round(4).astype(str),
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    )
    fig.update_layout(
        title="Global feature importance (mean absolute SHAP)",
        title_font_size=16,
        xaxis_title="Mean absolute SHAP value (log-odds)",
        yaxis_title=None,
        coloraxis_showscale=False,
        height=720,
        margin=dict(t=60, b=20, l=20, r=80),
    )
    fig.update_xaxes(range=[0, df["mean_abs_shap"].max() * 1.15])
    return fig


def build_beeswarm_figure(
    shap_values: np.ndarray,
    X_background: np.ndarray,
    feature_names: list,
    mean_abs_shap: np.ndarray,
) -> go.Figure:
    """SHAP summary plot: per-sample SHAP values, coloured by feature value."""
    order = np.argsort(-mean_abs_shap)
    ordered_names = [feature_names[j] for j in order]

    rows = []
    for j in order:
        feature_shap = shap_values[:, j]
        feature_vals = X_background[:, j]
        vmin, vmax = float(np.min(feature_vals)), float(np.max(feature_vals))
        span = vmax - vmin
        rel = (
            (feature_vals - vmin) / span if span > 0 else np.full_like(feature_vals, 0.5)
        )
        for i in range(len(feature_shap)):
            rows.append(
                {
                    "feature": feature_names[j],
                    "shap_value": float(feature_shap[i]),
                    "relative_value": float(rel[i]),
                }
            )

    df = pd.DataFrame(rows)
    fig = px.strip(
        df,
        x="shap_value",
        y="feature",
        color="relative_value",
        category_orders={"feature": ordered_names},
        hover_data={"shap_value": ":.4f", "relative_value": ":.2f"},
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    fig.add_vline(x=0, line_color="#555555", line_width=1)
    fig.update_layout(
        title="SHAP summary — per-sample contributions by feature",
        title_font_size=16,
        xaxis_title="SHAP value (log-odds) — right pushes toward diabetes",
        yaxis_title=None,
        height=720,
        margin=dict(t=60, b=20, l=20, r=20),
        coloraxis_colorbar=dict(
            title="Relative<br>feature value",
            tickvals=[0, 1],
            ticktext=["Low", "High"],
        ),
    )
    return fig


def build_contribution_figure(
    shap_row: np.ndarray,
    feature_names: list,
    raw_values: Dict[str, Any],
) -> go.Figure:
    """Feature contributions for one prediction (waterfall-style bar chart)."""
    # explain_instance output can be a 2-D (features x classes) row in some
    # scikit-learn/legacy layouts. Ravel to 1-D and re-align the feature name
    # list so the per-feature DataFrame and plotly axes stay 1-D.
    shap_row = np.asarray(shap_row).ravel()
    n = min(len(shap_row), len(feature_names))
    feature_names = list(feature_names)[:n]
    shap_row = shap_row[:n]
    df = pd.DataFrame(
        {"feature": feature_names, "shap_value": shap_row}
    )
    df["label"] = df["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
    df["feature_display"] = df["feature"].map(
        lambda f: f"{FEATURE_LABELS.get(f, f)} = {_format_feature_value(f, raw_values.get(f))}"
    )
    df = df.sort_values("shap_value", key=lambda s: s.abs(), ascending=True)

    colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in df["shap_value"]]

    fig = go.Figure(
        go.Bar(
            x=df["shap_value"],
            y=df["label"],
            orientation="h",
            marker_color=colors,
            customdata=df["feature_display"],
            text=[f"{v:+.3f}" for v in df["shap_value"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                "SHAP contribution: %{x:+.4f} (log-odds)<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_color="#555555", line_width=1)
    fig.update_layout(
        title="Feature contributions for this prediction",
        title_font_size=16,
        xaxis_title=(
            "← pushes away from diabetes (blue)   |   "
            "pushes toward diabetes (red) →"
        ),
        yaxis_title=None,
        height=720,
        margin=dict(t=60, b=20, l=20, r=80),
        showlegend=False,
    )
    return fig


# --------------------------------------------------------------------------- #
# Input form (mirrors the Individual Risk page inputs)
# --------------------------------------------------------------------------- #
def _binary_selectbox(feature: str, default: int, key: str) -> int:
    options = FEATURE_OPTIONS[feature]
    return int(
        st.selectbox(
            FEATURE_LABELS[feature],
            options=list(options.keys()),
            format_func=lambda x: options[x],
            index=list(options.keys()).index(default),
            key=key,
        )
    )


def build_prediction_inputs() -> Tuple[bool, Dict[str, Any]]:
    """Render the 21-feature input form and return (submitted, input_data)."""
    with st.form("xai_prediction_form"):
        st.markdown("### Patient information")
        st.caption(
            "Same 21 BRFSS features used by the Individual Risk page and the trained model."
        )

        st.markdown("#### 🩺 Health indicators")
        col1, col2 = st.columns(2)
        with col1:
            high_bp = _binary_selectbox("HighBP", 0, "xai_HighBP")
            high_chol = _binary_selectbox("HighChol", 0, "xai_HighChol")
            chol_check = _binary_selectbox("CholCheck", 1, "xai_CholCheck")
            stroke = _binary_selectbox("Stroke", 0, "xai_Stroke")
            heart_disease = _binary_selectbox("HeartDiseaseorAttack", 0, "xai_HeartDiseaseorAttack")
        with col2:
            diff_walk = _binary_selectbox("DiffWalk", 0, "xai_DiffWalk")
            gen_hlth = st.selectbox(
                FEATURE_LABELS["GenHlth"],
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: FEATURE_OPTIONS["GenHlth"][x],
                index=2,
                key="xai_GenHlth",
            )
            ment_hlth = st.number_input(
                FEATURE_LABELS["MentHlth"],
                min_value=FEATURE_RANGES["MentHlth"]["min"],
                max_value=FEATURE_RANGES["MentHlth"]["max"],
                value=FEATURE_RANGES["MentHlth"]["mean"],
                step=1,
                key="xai_MentHlth",
            )
            phys_hlth = st.number_input(
                FEATURE_LABELS["PhysHlth"],
                min_value=FEATURE_RANGES["PhysHlth"]["min"],
                max_value=FEATURE_RANGES["PhysHlth"]["max"],
                value=FEATURE_RANGES["PhysHlth"]["mean"],
                step=1,
                key="xai_PhysHlth",
            )

        st.markdown("#### 🏃 Lifestyle factors")
        col1, col2, col3 = st.columns(3)
        with col1:
            phys_activity = _binary_selectbox("PhysActivity", 1, "xai_PhysActivity")
            fruits = _binary_selectbox("Fruits", 1, "xai_Fruits")
        with col2:
            veggies = _binary_selectbox("Veggies", 1, "xai_Veggies")
            smoker = _binary_selectbox("Smoker", 0, "xai_Smoker")
        with col3:
            hvy_alcohol = _binary_selectbox("HvyAlcoholConsump", 0, "xai_HvyAlcoholConsump")

        st.markdown("#### 👤 Demographics")
        col1, col2, col3 = st.columns(3)
        with col1:
            sex = _binary_selectbox("Sex", 1, "xai_Sex")
            age = st.selectbox(
                FEATURE_LABELS["Age"],
                options=list(range(1, 14)),
                format_func=lambda x: FEATURE_OPTIONS["Age"][x],
                index=6,
                key="xai_Age",
            )
        with col2:
            education = st.selectbox(
                FEATURE_LABELS["Education"],
                options=[1, 2, 3, 4, 5, 6],
                format_func=lambda x: FEATURE_OPTIONS["Education"][x],
                index=3,
                key="xai_Education",
            )
            income = st.selectbox(
                FEATURE_LABELS["Income"],
                options=[1, 2, 3, 4, 5, 6, 7, 8],
                format_func=lambda x: FEATURE_OPTIONS["Income"][x],
                index=4,
                key="xai_Income",
            )
        with col3:
            bmi = st.number_input(
                FEATURE_LABELS["BMI"],
                min_value=FEATURE_RANGES["BMI"]["min"],
                max_value=FEATURE_RANGES["BMI"]["max"],
                value=FEATURE_RANGES["BMI"]["mean"],
                step=0.1,
                format="%.1f",
                key="xai_BMI",
            )

        st.markdown("#### 🏥 Healthcare access")
        col1, col2 = st.columns(2)
        with col1:
            any_healthcare = _binary_selectbox("AnyHealthcare", 1, "xai_AnyHealthcare")
        with col2:
            no_doc_cost = _binary_selectbox("NoDocbcCost", 0, "xai_NoDocbcCost")

        submitted = st.form_submit_button(
            "🔍 Explain this prediction", type="primary"
        )

    input_data = {
        "HighBP": high_bp,
        "HighChol": high_chol,
        "CholCheck": chol_check,
        "Smoker": smoker,
        "Stroke": stroke,
        "HeartDiseaseorAttack": heart_disease,
        "PhysActivity": phys_activity,
        "Fruits": fruits,
        "Veggies": veggies,
        "HvyAlcoholConsump": hvy_alcohol,
        "AnyHealthcare": any_healthcare,
        "NoDocbcCost": no_doc_cost,
        "DiffWalk": diff_walk,
        "Sex": sex,
        "GenHlth": gen_hlth,
        "Age": age,
        "Education": education,
        "Income": income,
        "BMI": bmi,
        "MentHlth": ment_hlth,
        "PhysHlth": phys_hlth,
    }
    return submitted, input_data


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def _show_disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer">
            <strong>Important limitation:</strong> this page explains how the trained
            machine-learning model behaves — it does <strong>not</strong> establish medical
            causation and is <strong>not</strong> a diagnosis. SHAP values describe the model's
            internal associations, not clinical or causal effects.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_individual_explanation(input_data: Dict[str, Any]) -> None:
    predictor = _load_predictor_cached()
    global_data = _load_global_explanations()
    explainer: ModelExplainer = global_data["explainer"]

    try:
        probability = predictor.predict_proba(input_data)
        pred_class = predictor.predict(input_data)
        risk_category = predictor.get_risk_category(probability)
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Prediction failed: {exc}")
        return

    st.subheader("📋 Prediction")
    with st.container(horizontal=True):
        st.metric("Predicted probability", f"{probability:.1%}", border=True)
        st.metric(
            "Predicted class",
            "Diabetes" if pred_class == 1 else "No diabetes",
            border=True,
        )
        st.metric("Risk category", risk_category, border=True)

    try:
        row = pd.DataFrame(
            [[input_data[f] for f in BRFSS_FEATURES]], columns=BRFSS_FEATURES
        )
        X_instance = predictor.preprocessor.transform(row)
        shap_row = np.asarray(explainer.explain_instance(X_instance))[0]
        # explain_instance can report a (features, classes) or per-sample
        # multi-class slice; keep the diabetes (positive) class as 1-D so the
        # contribution figure and per-feature DataFrame stay well-formed.
        shap_row = np.asarray(shap_row)
        if shap_row.ndim > 1:
            shap_row = shap_row[..., 1] if shap_row.shape[-1] > 1 else shap_row[..., 0]
        shap_row = shap_row.ravel()
    except Exception as exc:  # pragma: no cover - defensive
        st.error(f"Could not compute SHAP explanation: {exc}")
        return

    st.subheader("🧩 Feature contributions")
    st.plotly_chart(
        build_contribution_figure(shap_row, global_data["feature_names"], input_data)
    )

    base = _base_value(explainer)
    st.caption(
        f"Red bars contribute **toward** the diabetes class; blue bars contribute "
        f"**away** from it. Values are in log-odds (raw model output). "
        f"Base model output for the diabetes class: {base:+.3f}."
    )

    contrib = pd.DataFrame(
        {"feature": global_data["feature_names"], "shap_value": shap_row}
    ).sort_values("shap_value")

    top_toward = contrib.iloc[-1]
    top_away = contrib.iloc[0]
    toward_label = FEATURE_LABELS.get(top_toward["feature"], top_toward["feature"])
    away_label = FEATURE_LABELS.get(top_away["feature"], top_away["feature"])

    st.markdown(
        f"- **Largest push toward diabetes:** {toward_label} "
        f"(SHAP {top_toward['shap_value']:+.3f})."
    )
    st.markdown(
        f"- **Largest push away from diabetes:** {away_label} "
        f"(SHAP {top_away['shap_value']:+.3f})."
    )
    st.markdown(
        "- These are **model explanations/associations** — how the model weighted the inputs "
        "for this individual — not causal effects and not a medical assessment."
    )


def explainability_page() -> None:
    """Render the Explainable AI page."""
    st.markdown('<div class="main-header">🔍 Explainable AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">SHAP explanations for the trained diabetes risk model</div>',
        unsafe_allow_html=True,
    )
    _show_disclaimer()

    global_data = _load_global_explanations()
    feature_names = global_data["feature_names"]

    st.subheader("🧠 Model")
    with st.container(horizontal=True):
        st.metric("Model", type(global_data["model"]).__name__, border=True)
        st.metric("Features", str(len(feature_names)), border=True)
        st.metric("Explanation method", "TreeSHAP", border=True)
        st.metric(
            "Background samples", str(global_data["X_background"].shape[0]), border=True
        )
    st.caption(
        "Explains `models/best_model.joblib` with `models/preprocessor.joblib`. "
        f"The {len(feature_names)} features are the exact BRFSS features used by the model."
    )
    with st.expander("View the 21 model features"):
        st.write(", ".join(feature_names))

    st.markdown("---")
    st.subheader("📊 Global feature importance")
    st.plotly_chart(
        build_importance_figure(global_data["mean_abs_shap"], feature_names)
    )
    st.caption(
        "Mean absolute SHAP value per feature across background samples — a larger value "
        "means the feature moves the model's output more, on average. It shows model reliance, "
        "not causal importance."
    )
    with st.expander("View feature importance table"):
        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Mean |SHAP|": global_data["mean_abs_shap"].round(4),
            }
        ).sort_values("Mean |SHAP|", ascending=False)
        st.dataframe(importance_df, hide_index=True)

    st.markdown("---")
    st.subheader("🐝 SHAP summary")
    st.plotly_chart(
        build_beeswarm_figure(
            global_data["shap_values"],
            global_data["X_background"],
            feature_names,
            global_data["mean_abs_shap"],
        )
    )
    st.caption(
        "Each dot is one background sample. Horizontal position is its SHAP value "
        "(right = toward diabetes, left = away). Colour is the feature's value relative to "
        "its own range (red = high, blue = low)."
    )

    st.markdown("---")
    st.subheader("🎯 Explain a single prediction")
    st.caption(
        "Fill in the patient information and click **Explain this prediction** to see which "
        "features pushed the model toward or away from the diabetes class for that individual."
    )
    submitted, input_data = build_prediction_inputs()
    if submitted:
        _render_individual_explanation(input_data)

    st.markdown("---")
    st.subheader("📖 How to read these explanations")
    st.markdown(
        "- **Positive contribution (red):** the feature pushes the model's output **toward** "
        "the diabetes class (increases the predicted probability).\n"
        "- **Negative contribution (blue):** the feature pushes the model's output **away** "
        "from the diabetes class (decreases the predicted probability).\n"
        "- SHAP values are in **log-odds** (raw model output) units and are additive: "
        "base value + sum of contributions = the model's raw output for the diabetes class.\n"
        "- These are **model explanations/associations**, not causal effects. A feature being "
        "important to the model does not mean changing it would change a person's risk."
    )

    st.warning(
        "**Limitation:** SHAP explains the behaviour of the trained model only. It does not "
        "establish medical causation, does not provide a diagnosis, and should not be used "
        "for clinical decisions. The model is trained on self-reported BRFSS 2015 survey data."
    )

    st.info(
        "No model is trained on this page. The SHAP explainer is computed once and cached "
        "(in-session and at `models/best_model_shap_explainer.joblib`)."
    )