"""What-If Simulation page — explore how feature changes move predicted risk.

Uses the same trained Module 1 model (``models/best_model.joblib``) and
preprocessor (``models/preprocessor.joblib``) as the Individual Risk page,
through the shared ``DiabetesRiskPredictor`` pipeline in ``src.predict``.

The model is never retrained here. The page shows a baseline set of the same
21 BRFSS features and an editable scenario, then compares the predicted
diabetes risk probability between the two. The output is a **model
simulation** — how the trained model re-weights the inputs — not a medical
outcome nor a diagnosis.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import streamlit as st

from src.predict import (
    BRFSS_FEATURES,
    FEATURE_LABELS,
    FEATURE_OPTIONS,
    FEATURE_RANGES,
)
from src.utils.model_loader import load_predictor


# Default widget index per feature, matching the Individual Risk page defaults.
_DEFAULT_INDEX = {
    "HighBP": 0,
    "HighChol": 0,
    "CholCheck": 1,
    "Smoker": 0,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "DiffWalk": 0,
    "Sex": 1,
    "GenHlth": 2,
    "Age": 6,
    "Education": 3,
    "Income": 4,
}

# Numeric step per continuous feature.
_NUMERIC_STEP = {
    "BMI": 0.1,
    "MentHlth": 1,
    "PhysHlth": 1,
}


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading trained model and preprocessor...")
def _load_predictor_cached():
    return load_predictor()


# --------------------------------------------------------------------------- #
# Input helpers (mirror the Individual Risk page inputs)
# --------------------------------------------------------------------------- #
def _render_feature_inputs(prefix: str) -> Dict[str, Any]:
    """Render the 21 BRFSS feature inputs under a widget-key ``prefix``."""
    data: Dict[str, Any] = {}
    for feature in BRFSS_FEATURES:
        key = f"{prefix}_{feature}"
        options = FEATURE_OPTIONS.get(feature)
        if options is not None and len(options) == 2:
            data[feature] = int(
                st.selectbox(
                    FEATURE_LABELS[feature],
                    options=[0, 1],
                    format_func=lambda x, f=feature: FEATURE_OPTIONS[f][x],
                    index=_DEFAULT_INDEX.get(feature, 0),
                    key=key,
                )
            )
        elif options is not None:
            option_keys = list(options.keys())
            data[feature] = st.selectbox(
                FEATURE_LABELS[feature],
                options=option_keys,
                format_func=lambda x, f=feature: FEATURE_OPTIONS[f][x],
                index=_DEFAULT_INDEX.get(feature, len(option_keys) // 2),
                key=key,
            )
        else:
            rng = FEATURE_RANGES[feature]
            data[feature] = st.number_input(
                FEATURE_LABELS[feature],
                min_value=rng["min"],
                max_value=rng["max"],
                value=rng["mean"],
                step=_NUMERIC_STEP.get(feature, 1),
                format="%.1f" if feature == "BMI" else "%d",
                key=key,
            )
    return data


def _format_binary(value: Any) -> str:
    if value in (0, 1):
        return "No" if value == 0 else "Yes"
    return str(value)


def _format_feature_value(feature: str, value: Any) -> str:
    options = FEATURE_OPTIONS.get(feature)
    if options is not None and value in options:
        return f"{options[value]} ({value})"
    if feature == "Sex" and value in (0, 1):
        return "Female" if value == 0 else "Male"
    return str(value)


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def _simulation_disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer">
            <strong>⚠️ Model simulation — not a medical outcome:</strong> this page
            answers <em>"how would the trained model's predicted risk change if the
            inputs change?"</em> It does <strong>NOT</strong> predict an actual medical
            outcome, diagnose diabetes, or establish causation. The change in
            probability reflects how the model re-weights the 21 BRFSS features — treat
            it as a simulation on a statistical screening model, and consult a
            healthcare provider for medical decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )


def what_if_simulation_page() -> None:
    """Render the What-If Simulation page."""
    st.markdown(
        '<div class="main-header">🧪 What-If Simulation</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">Explore how changing inputs moves the model\'s predicted risk</div>',
        unsafe_allow_html=True,
    )
    _simulation_disclaimer()

    try:
        predictor = _load_predictor_cached()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        st.stop()

    with st.container(horizontal=True):
        st.metric("Model", type(predictor.model).__name__, border=True)
        st.metric("Features", str(len(BRFSS_FEATURES)), border=True)
        st.metric("Pipeline", "Individual Risk (BRFSS 2015)", border=True)

    st.caption(
        "Uses `models/best_model.joblib` (Random Forest) and "
        "`models/preprocessor.joblib` — the same pipeline as the Individual Risk page. "
        "No model is retrained on this page."
    )

    with st.form("what_if_simulation_form"):
        st.markdown("### Baseline vs Scenario")
        st.caption(
            "Baseline is the starting patient profile. Edit Scenario to simulate the effect "
            "of changes (e.g. lowering BMI, quitting smoking). Both use the same 21 BRFSS features."
        )
        col_base, col_scen = st.columns(2)

        with col_base:
            st.subheader("📋 Baseline inputs")
            baseline = _render_feature_inputs("wi_base")
        with col_scen:
            st.subheader("🔁 Scenario inputs")
            scenario = _render_feature_inputs("wi_scen")

        submitted = st.form_submit_button("🧪 Run simulation", type="primary")

    if not submitted:
        st.info(
            "Fill in both profiles and click **Run simulation** to compare the two "
            "predicted risks and see how changing the scenario inputs moves the model's "
            "prediction relative to the baseline."
        )
        return

    # Validate both input dictionaries.
    missing = [f for f in BRFSS_FEATURES if f not in baseline or baseline[f] is None]
    if missing:
        st.error(f"Missing baseline inputs: {missing}")
        return
    missing_scen = [
        f for f in BRFSS_FEATURES if f not in scenario or scenario[f] is None
    ]
    if missing_scen:
        st.error(f"Missing scenario inputs: {missing_scen}")
        return

    try:
        base_prob = predictor.predict_proba(baseline)
        scen_prob = predictor.predict_proba(scenario)
    except Exception as exc:
        st.error(f"Simulation failed: {exc}")
        return

    base_class = predictor.predict(baseline)
    scen_class = predictor.predict(scenario)
    base_cat = predictor.get_risk_category(base_prob)
    scen_cat = predictor.get_risk_category(scen_prob)

    delta_pp = (scen_prob - base_prob) * 100

    st.markdown("---")
    st.subheader("📊 Simulation results")

    with st.container(horizontal=True):
        st.metric(
            "Baseline probability",
            f"{base_prob:.1%}",
            f"{base_cat}",
            border=True,
            delta_color="off",
        )
        st.metric(
            "Scenario probability",
            f"{scen_prob:.1%}",
            f"{scen_cat}",
            border=True,
            delta_color="off",
        )
        direction = "increase" if delta_pp >= 0 else "decrease"
        st.metric(
            "Difference (scenario − baseline)",
            f"{delta_pp:+.1f} pp",
            f"{direction} in predicted risk",
            border=True,
            delta_color="inverse",
        )

    st.markdown(
        f"- **Baseline:** {base_cat} ({base_prob:.1%}) — predicted "
        f"**{'Diabetes' if base_class == 1 else 'No diabetes'}**.\n"
        f"- **Scenario:** {scen_cat} ({scen_prob:.1%}) — predicted "
        f"**{'Diabetes' if scen_class == 1 else 'No diabetes'}**.\n"
        f"- **Change:** the scenario moves the predicted probability "
        f"{'**up** by' if delta_pp >= 0 else '**down** by'} "
        f"**{abs(delta_pp):.1f} percentage points** "
        f"({'toward' if delta_pp >= 0 else 'away from'} the diabetes class)."
    )
    st.markdown(
        f"- **Largest driver of the change:** the features whose values differ between "
        f"the baseline and scenario are "
        f"**{', '.join([FEATURE_LABELS.get(f, f) for f in BRFSS_FEATURES if baseline[f] != scenario[f]] or ['none'])}**."
    )

    with st.expander("View feature-level comparison"):
        comp = []
        for f in BRFSS_FEATURES:
            base_v = _format_feature_value(f, baseline[f])
            scen_v = _format_feature_value(f, scenario[f])
            comp.append(
                {
                    "Feature": FEATURE_LABELS.get(f, f),
                    "Baseline": base_v,
                    "Scenario": scen_v,
                    "Changed": "\u2705" if baseline[f] != scenario[f] else "",
                }
            )
        import pandas as pd

        st.dataframe(pd.DataFrame(comp), hide_index=True, width="stretch")

    st.warning(
        "**Reminder:** This is a simulation on a machine-learning screening model trained "
        "on self-reported CDC BRFSS 2015 data. A change in predicted probability reflects the "
        "model's associations, not a guaranteed medical outcome and not causation."
    )