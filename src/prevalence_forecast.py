"""Prevalence Forecast page (Module 3) for the Streamlit app.

Loads the saved Module 3 outputs and results — the WHO India prevalence
series (``data/processed/india_forecasting_data.csv``) and the saved
ARIMA forecast artifacts under ``outputs/reports/`` — and renders them
interactively.

Nothing is retrained and no new forecast is calculated on this page: every
value and metric shown is read from the existing saved outputs.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils.model_loader import load_forecast_data

REPORTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "reports"

TEST_METRICS_FILE = "arima_test_metrics.csv"
VALIDATION_COMPARISON_FILE = "model_comparison_validation.csv"

HISTORY_START = 1990
HISTORY_END = 2022


def _model_label(order: str) -> str:
    """Normalise a saved ARIMA order string, e.g. '(2, 2, 0)' -> 'ARIMA(2,2,0)'."""
    return f"ARIMA{order.replace(' ', '')}"


@functools.lru_cache(maxsize=1)
def load_forecast_reports() -> Dict[str, pd.DataFrame]:
    """Load all saved Module 3 outputs used by the page."""
    data = load_forecast_data()
    data["test_metrics"] = pd.read_csv(REPORTS_DIR / TEST_METRICS_FILE)
    data["validation_comparison"] = pd.read_csv(REPORTS_DIR / VALIDATION_COMPARISON_FILE)
    return data


def build_combined_figure(data: Dict[str, pd.DataFrame]) -> go.Figure:
    """Historical WHO estimates (1990-2022) + saved ARIMA forecast (2023-2032)."""
    hist = data["india_series"]
    fut = data["future_forecast"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist["year"],
            y=hist["prevalence_pct"],
            mode="lines+markers",
            name="WHO estimate (historical)",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hist["year"],
            y=hist["high_ci"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=hist["year"],
            y=hist["low_ci"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(31,119,180,0.12)",
            name="WHO 95% uncertainty interval",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=fut["year"],
            y=fut["predicted"],
            mode="lines+markers",
            name="ARIMA forecast (model)",
            line=dict(color="#d62728", width=2, dash="dash"),
            marker=dict(size=6, symbol="square"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fut["year"],
            y=fut["ci_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fut["year"],
            y=fut["ci_lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(214,39,40,0.12)",
            name="Forecast 95% prediction interval",
            hoverinfo="skip",
        )
    )

    fig.add_vline(x=2022.5, line_dash="dot", line_color="#7f7f7f", line_width=1)
    fig.add_annotation(
        x=2003, y=1.0, xref="x", yref="paper",
        text="WHO historical estimates (1990–2022)",
        showarrow=False, font=dict(size=12, color="#1f77b4"),
    )
    fig.add_annotation(
        x=2028, y=1.0, xref="x", yref="paper",
        text="Model forecast (2023–2032)",
        showarrow=False, font=dict(size=12, color="#d62728"),
    )

    fig.update_layout(
        title="India diabetes prevalence: WHO estimates and ARIMA forecast",
        title_font_size=16,
        xaxis_title="Year",
        yaxis_title="Age-standardised diabetes prevalence (%)",
        height=520,
        margin=dict(t=70, b=30, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0),
    )
    return fig


def build_test_figure(data: Dict[str, pd.DataFrame]) -> go.Figure:
    """Held-out test period (2016-2022): WHO actual vs ARIMA predicted."""
    t = data["test_predictions"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t["year"], y=t["actual"],
            mode="lines+markers",
            name="WHO actual",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t["year"], y=t["ci_upper"],
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t["year"], y=t["ci_lower"],
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(214,39,40,0.12)",
            name="95% prediction interval",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=t["year"], y=t["predicted"],
            mode="lines+markers",
            name="ARIMA predicted",
            line=dict(color="#d62728", width=2, dash="dash"),
            marker=dict(size=8, symbol="square"),
        )
    )

    fig.update_layout(
        title="Held-out test period (2016–2022): WHO actual vs ARIMA predicted",
        title_font_size=16,
        xaxis_title="Year",
        yaxis_title="Age-standardised diabetes prevalence (%)",
        height=440,
        margin=dict(t=60, b=30, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0),
        xaxis=dict(tickmode="array", tickvals=t["year"].tolist()),
    )
    return fig


def build_validation_comparison_figure(data: Dict[str, pd.DataFrame]) -> go.Figure:
    """Validation-period MAE comparison for model selection context."""
    cmp = data["validation_comparison"].sort_values("mae", ascending=True)
    colors = ["#2ca02c" if m == "arima" else "#1f77b4" for m in cmp["model"]]

    fig = go.Figure(
        go.Bar(
            x=cmp["model"],
            y=cmp["mae"],
            marker_color=colors,
            text=cmp["mae"].round(3).astype(str),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Validation-period (2008–2015) MAE by model — ARIMA selected",
        title_font_size=16,
        xaxis_title=None,
        yaxis_title="MAE (percentage points)",
        height=360,
        margin=dict(t=60, b=30, l=20, r=20),
        showlegend=False,
        bargap=0.4,
    )
    fig.update_yaxes(range=[0, cmp["mae"].max() * 1.15])
    return fig


def _test_metrics_summary(data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    row = data["test_metrics"].iloc[0]
    return {
        "model": _model_label(str(row["order"])),
        "mae": f"{row['mae']:.4f}",
        "rmse": f"{row['rmse']:.4f}",
        "mape": f"{row['mape']:.2f}%",
    }


MODEL_INTERPRETATION: List[str] = [
    "**What the forecast represents:** the dashed red line extrapolates the WHO "
    "age-standardised diabetes prevalence series for India (adults 18+, both sexes, "
    "1990–2022) into 2023–2032 using an ARIMA(2,2,0) time-series model. Each point is "
    "the model's expected prevalence (%) for that year — not a new measurement.",
    "**It is a model estimate:** the blue WHO historical values are themselves modeled "
    "WHO estimates (not direct survey measurements), and the forecast extends that series "
    "by statistical extrapolation. It is not a clinician's assessment and it is not a "
    "guarantee of future data.",
    "**Model choice:** ARIMA(2,2,0) produced the lowest validation-period (2008–2015) MAE "
    "among the four candidate models (Naive, Linear trend, ARIMA, XGBoost). It was then "
    "evaluated on the held-out test period (2016–2022), which was not used to select the "
    "model — test MAE 0.465, RMSE 0.586 (percentage points), MAPE 2.17%.",
    "**Short annual series:** only 33 annual observations (1990–2022) underpin the model. "
    "Extrapolating that short series ten years into the future is inherently uncertain, "
    "even though the forecast line looks smooth.",
    "**Modeled WHO inputs:** the input series consists of WHO modeled estimates with their "
    "own 95% uncertainty intervals, so the forecast inherits that input uncertainty in "
    "addition to its own prediction interval.",
    "**Univariate trend model:** ARIMA uses only past prevalence values — it includes no "
    "covariates (e.g. changing obesity, ageing, screening effort). An I(2)-differenced model "
    "can keep projecting an accelerating trend well beyond the observed historical range.",
    "**Read as a scenario, not a certainty:** the forecast is a projection from a fitted "
    "trend whose confidence intervals widen through 2032. Use it for planning context, not "
    "as a precise prediction.",
]


def _show_disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer">
            <strong>Interpret with care:</strong> the forecast on this page is a
            <strong>statistical model projection</strong> built from WHO modeled estimates.
            It is <em>not</em> a medical or epidemiological pronouncement, and real future
            prevalence may differ materially.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_data_table(data: Dict[str, pd.DataFrame]) -> None:
    fut = data["future_forecast"]
    future_table = pd.DataFrame(
        {
            "Year": fut["year"].astype(int),
            "Predicted prevalence (%)": fut["predicted"].round(2),
            "95% CI lower (%)": fut["ci_lower"].round(2),
            "95% CI upper (%)": fut["ci_upper"].round(2),
        }
    )
    st.markdown("**Future forecast (2023–2032)** — `outputs/reports/arima_future_forecast.csv`")
    st.dataframe(future_table, hide_index=True)

    t = data["test_predictions"]
    test_table = pd.DataFrame(
        {
            "Year": t["year"].astype(int),
            "WHO actual (%)": t["actual"].round(2),
            "ARIMA predicted (%)": t["predicted"].round(2),
            "Error (pp)": t["error"].round(3),
        }
    )
    st.markdown("**Test period (2016–2022) actual vs predicted** — `outputs/reports/arima_test_predictions.csv`")
    st.dataframe(test_table, hide_index=True)


def prevalence_forecast_page() -> None:
    """Render the Prevalence Forecast (Module 3) page."""
    st.markdown('<div class="main-header">🔮 Prevalence Forecast</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">India diabetes prevalence — WHO historical estimates & ARIMA forecast</div>',
        unsafe_allow_html=True,
    )
    _show_disclaimer()

    data = load_forecast_reports()
    summary = _test_metrics_summary(data)
    fut = data["future_forecast"]

    st.subheader("📌 Forecast summary")
    with st.container(horizontal=True):
        st.metric("Selected model", summary["model"], border=True)
        st.metric("Test MAE", summary["mae"], border=True)
        st.metric("Test RMSE", summary["rmse"], border=True)
        st.metric(
            "Forecast period",
            f"{int(fut['year'].min())}–{int(fut['year'].max())}",
            border=True,
        )
    st.caption(
        "Test metrics are the saved held-out results (2016–2022) for the selected model — "
        "`outputs/reports/arima_test_metrics.csv`."
    )

    st.markdown("---")
    st.subheader("📈 Historical estimates and forecast")
    st.plotly_chart(build_combined_figure(data))
    st.caption(
        "Solid blue line = WHO modeled historical estimates (1990–2022) with their 95% "
        "uncertainty interval; dashed red line = saved ARIMA forecast (2023–2032) with its "
        "95% prediction interval. The dotted vertical line separates WHO historical estimates "
        "from the model forecast. Values are shown exactly as saved."
    )

    st.markdown("---")
    st.subheader("🧪 Held-out test period (2016–2022)")
    st.plotly_chart(build_test_figure(data))
    st.caption(
        "The test years were held out of model selection. The ARIMA(2,2,0) forecast for "
        "2016–2022 is compared against the WHO estimates for the same years."
    )

    st.markdown("---")
    st.subheader("⚖️ Why ARIMA? — Validation comparison")
    st.plotly_chart(build_validation_comparison_figure(data))
    st.caption(
        "Validation-period MAE (2008–2015) across the four candidate models from "
        "`outputs/reports/model_comparison_validation.csv`. ARIMA had the lowest error and "
        "was selected for the final forecast."
    )

    with st.expander("View saved forecast data tables", expanded=False):
        _render_data_table(data)

    st.markdown("---")
    st.subheader("💡 Model interpretation")
    for item in MODEL_INTERPRETATION:
        st.markdown(f"- {item}")

    st.warning(
        "Note: the saved forecast (`arima_future_forecast.csv`) projects 18.7% for 2023, "
        "below the 2022 WHO estimate of 22.6%. Values are displayed exactly as saved; "
        "interpret near-term values alongside the uncertainty band and the recent observed trend."
    )

    st.info(
        "No model is trained and no new forecast is computed on this page. All data come from "
        "`data/processed/india_forecasting_data.csv` and the saved Module 3 reports under "
        "`outputs/reports/`."
    )