"""Population Analytics page (Module 2) for the Streamlit app.

Loads the pre-computed Module 2 reports saved under ``outputs/reports/`` by
:mod:`src.population_analysis` and renders them interactively.

Only the saved CSV reports are read on this page; the raw BRFSS dataset is
never loaded here.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.population_analysis import VALUE_LABELS
from src.utils.model_loader import load_population_data

REPORTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "reports"

# (report key inside load_population_data, feature column in the report CSV)
DIMENSIONS = [
    ("Age", "Age"),
    ("BMI category", "BMI_category"),
    ("Sex", "Sex"),
    ("GenHlth", "GenHlth"),
    ("PhysActivity", "PhysActivity"),
    ("HighBP", "HighBP"),
    ("HighChol", "HighChol"),
    ("Education", "Education"),
    ("Income", "Income"),
]

DIMENSION_TITLES = {
    "Age": "Diabetes prevalence by age group",
    "BMI category": "Diabetes prevalence by BMI category",
    "Sex": "Diabetes prevalence by sex",
    "GenHlth": "Diabetes prevalence by self-reported general health",
    "PhysActivity": "Diabetes prevalence by physical activity",
    "HighBP": "Diabetes prevalence by high blood pressure",
    "HighChol": "Diabetes prevalence by high cholesterol",
    "Education": "Diabetes prevalence by education level",
    "Income": "Diabetes prevalence by income level",
}

TAB_LABELS = {
    "Age": "Age",
    "BMI category": "BMI",
    "Sex": "Sex",
    "GenHlth": "General health",
    "PhysActivity": "Physical activity",
    "HighBP": "High BP",
    "HighChol": "High cholesterol",
    "Education": "Education",
    "Income": "Income",
}

BMI_ORDER = [
    "Underweight",
    "Normal",
    "Overweight",
    "Obese Class I",
    "Obese Class II",
    "Obese Class III",
]


def _apply_labels(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Attach human-readable category labels to a prevalence report table."""
    df = df.copy()
    if feature == "BMI_category":
        df["category"] = df[feature]
        return df
    df[feature] = df[feature].astype(int)
    labels = VALUE_LABELS.get(feature)
    df["category"] = df[feature].map(labels) if labels else df[feature]
    return df


@functools.lru_cache(maxsize=1)
def load_module2_reports() -> Dict[str, pd.DataFrame]:
    """Load and enrich every Module 2 report used by the page."""
    raw = load_population_data()

    reports: Dict[str, pd.DataFrame] = {}
    for key, feature in DIMENSIONS:
        df = _apply_labels(raw[key], feature)
        if feature == "BMI_category":
            rank = {label: i for i, label in enumerate(BMI_ORDER)}
            df["_sort"] = df[feature].map(rank)
            df = df.sort_values("_sort").drop(columns="_sort")
        else:
            df = df.sort_values(feature)
        reports[key] = df

    reports["overall"] = raw["overall prevalence"]
    reports["association"] = raw["association measures"]
    return reports


def build_prevalence_figure(df: pd.DataFrame, title: str, ymax_pad: float = 0.15) -> go.Figure:
    """Build an interactive bar chart of prevalence by category."""
    fig = px.bar(
        df,
        x="category",
        y="prevalence_pct",
        color="prevalence_pct",
        color_continuous_scale="Blues",
        custom_data=["sample_count", "diabetes_cases"],
        text=df["prevalence_pct"].round(2).astype(str) + "%",
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Prevalence: %{y:.2f}%<br>"
            "Sample: %{customdata[0]:,}<br>"
            "Diabetes cases: %{customdata[1]:,}<extra></extra>"
        ),
    )
    fig.update_layout(
        title=title,
        title_font_size=16,
        xaxis_title=None,
        yaxis_title="Diabetes prevalence (%)",
        coloraxis_showscale=False,
        height=420,
        margin=dict(t=60, b=20, l=20, r=20),
        bargap=0.35,
    )
    fig.update_yaxes(range=[0, df["prevalence_pct"].max() * (1 + ymax_pad)])
    return fig


def build_association_figure(assoc: pd.DataFrame) -> go.Figure:
    """Build a horizontal bar chart of Cramer's V association strength."""
    df = assoc.sort_values("cramers_v").copy()
    fig = px.bar(
        df,
        x="cramers_v",
        y="feature_label",
        orientation="h",
        color="cramers_v",
        color_continuous_scale="Reds",
        text=df["cramers_v"].round(3).astype(str),
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Cramér's V: %{x:.3f}<br>"
            "p-value: %{customdata}<extra></extra>"
        ),
        customdata=df["p_value"].round(4),
    )
    fig.update_layout(
        title="Association strength between features and diabetes (Cramér's V)",
        title_font_size=16,
        xaxis_title="Cramér's V",
        yaxis_title=None,
        coloraxis_showscale=False,
        height=520,
        margin=dict(t=60, b=20, l=20, r=60),
    )
    fig.update_xaxes(range=[0, df["cramers_v"].max() * 1.12])
    return fig


def _minmax(df: pd.DataFrame, column: str = "prevalence_pct") -> tuple:
    """Return (lowest, highest) rows of a prevalence report."""
    lo = df.loc[df[column].idxmin()]
    hi = df.loc[df[column].idxmax()]
    return lo, hi


def _row_by_category(df: pd.DataFrame, category: str) -> pd.Series:
    """Return the first row matching a category label."""
    return df[df["category"] == category].iloc[0]


def compute_key_findings(reports: Dict[str, pd.DataFrame]) -> List[str]:
    """Summarise the key findings using only the saved Module 2 values."""
    findings: List[str] = []

    overall = reports["overall"]
    total = int(overall["total_sample"].iloc[0])
    cases = int(overall["diabetes_cases"].iloc[0])
    prev = float(overall["prevalence_pct"].iloc[0])
    findings.append(
        f"**Overall prevalence** in the BRFSS 2015 sample is **{prev:.1f}%** "
        f"({cases:,} diabetes cases out of {total:,} respondents)."
    )

    lo, hi = _minmax(reports["Age"])
    findings.append(
        f"**Age:** prevalence climbs steadily with age, from **{lo['prevalence_pct']:.1f}%** "
        f"(ages {lo['category']}) to a peak of **{hi['prevalence_pct']:.1f}%** (ages {hi['category']})."
    )

    lo, hi = _minmax(reports["BMI category"])
    findings.append(
        f"**BMI:** prevalence rises sharply with BMI, from **{lo['prevalence_pct']:.1f}%** "
        f"({lo['category']}) to **{hi['prevalence_pct']:.1f}%** ({hi['category']})."
    )

    highbp = reports["HighBP"].set_index("category")["prevalence_pct"]
    findings.append(
        f"**High blood pressure:** **{highbp['Yes']:.1f}%** among those reporting high BP "
        f"vs **{highbp['No']:.1f}%** among those without."
    )

    highchol = reports["HighChol"].set_index("category")["prevalence_pct"]
    findings.append(
        f"**High cholesterol:** **{highchol['Yes']:.1f}%** among those reporting high "
        f"cholesterol vs **{highchol['No']:.1f}%** among those without."
    )

    activity = reports["PhysActivity"].set_index("category")["prevalence_pct"]
    findings.append(
        f"**Physical activity:** **{activity['No']:.1f}%** among the physically inactive "
        f"vs **{activity['Yes']:.1f}%** among the physically active."
    )

    genhlth = reports["GenHlth"].set_index("category")["prevalence_pct"]
    findings.append(
        f"**General health:** **{genhlth['Poor']:.1f}%** among those rating their health 'Poor' "
        f"vs **{genhlth['Excellent']:.1f}%** among those rating it 'Excellent'."
    )

    income = reports["Income"]
    low_inc = _row_by_category(income, "<$10,000")
    high_inc = _row_by_category(income, "$75,000+")
    findings.append(
        f"**Income:** **{low_inc['prevalence_pct']:.1f}%** in the lowest income bracket "
        f"(<$10,000) vs **{high_inc['prevalence_pct']:.1f}%** in the highest ($75,000+)."
    )

    education = reports["Education"]
    lo_edu, hi_edu = _minmax(education)
    findings.append(
        f"**Education:** **{hi_edu['prevalence_pct']:.1f}%** among respondents with the "
        f"lowest education ({hi_edu['category']}) vs **{lo_edu['prevalence_pct']:.1f}%** "
        f"for college graduates."
    )

    sex = reports["Sex"].set_index("category")["prevalence_pct"]
    findings.append(
        f"**Sex:** **{sex['Male']:.1f}%** among males vs **{sex['Female']:.1f}%** among females."
    )

    assoc = reports["association"].sort_values("cramers_v", ascending=False)
    top1, top2 = assoc.iloc[0], assoc.iloc[1]
    findings.append(
        f"**Association strength:** the strongest associations with diabetes are "
        f"**{top1['feature_label']}** (Cramér's V = {top1['cramers_v']:.3f}) and "
        f"**{top2['feature_label']}** (Cramér's V = {top2['cramers_v']:.3f})."
    )

    findings.append(
        "These are **descriptive associations** from the BRFSS 2015 sample — "
        "they do **not** establish cause and effect."
    )

    return findings


LIMITATIONS = [
    "**Cross-sectional survey:** BRFSS 2015 captures a single point in time, so the "
    "associations above cannot establish causality or sequencing (e.g. whether obesity "
    "preceded diabetes).",
    "**Self-reported data:** all indicators (BP, cholesterol, health rating, income, BMI from "
    "height/weight) are self-reported and not clinically measured.",
    "**Unweighted analysis:** prevalence figures are computed on the unweighted sample and do "
    "not apply BRFSS survey weights, so they are not nationally representative estimates.",
    "**Descriptive only:** these are descriptive associations (Cramér's V), not adjusted for "
    "confounders such as age or sex.",
]


def _show_disclaimer() -> None:
    st.markdown(
        """
        <div class="disclaimer">
            <strong>Interpret with care:</strong> the figures on this page are
            <strong>descriptive, cross-sectional associations</strong> from the CDC BRFSS 2015
            survey. They indicate that a group is <em>more frequently</em> diabetic — not that
            the factor <em>causes</em> diabetes.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_dimension(reports: Dict[str, pd.DataFrame], key: str, feature: str) -> None:
    df = reports[key]
    st.plotly_chart(build_prevalence_figure(df, DIMENSION_TITLES[key]))
    table = pd.DataFrame(
        {
            "Category": df["category"].tolist(),
            "Respondents": df["sample_count"].tolist(),
            "Diabetes cases": df["diabetes_cases"].astype(int).tolist(),
            "Prevalence (%)": df["prevalence_pct"].round(2).tolist(),
        }
    )
    st.dataframe(table, hide_index=True)


def population_analytics_page() -> None:
    """Render the Population Analytics (Module 2) page."""
    st.markdown('<div class="main-header">📊 Population Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Diabetes prevalence by demographics and risk factors — CDC BRFSS 2015</div>',
        unsafe_allow_html=True,
    )
    _show_disclaimer()

    reports = load_module2_reports()
    overall = reports["overall"]

    st.subheader("📈 Overall diabetes prevalence")
    with st.container(horizontal=True):
        st.metric("Overall prevalence", f"{overall['prevalence_pct'].iloc[0]:.2f}%", border=True)
        st.metric("Diabetes cases", f"{int(overall['diabetes_cases'].iloc[0]):,}", border=True)
        st.metric("Sample size", f"{int(overall['total_sample'].iloc[0]):,}", border=True)
    st.caption(
        "Prevalence = diabetes cases ÷ sample. BRFSS 2015 cleaned sample "
        "(duplicates removed). Source: outputs/reports/overall_prevalence.csv"
    )

    st.markdown("---")
    st.subheader("📊 Prevalence by subgroup")

    tabs = st.tabs([TAB_LABELS[key] for key, _ in DIMENSIONS])
    for tab, (key, feature) in zip(tabs, DIMENSIONS):
        with tab:
            _render_dimension(reports, key, feature)

    st.markdown("---")
    st.subheader("🔗 Association strength (Cramér's V)")
    st.caption(
        "Cramér's V measures how strongly each feature is associated with diabetes status "
        "(0 = no association, 1 = perfect association). Descriptive only — not causal."
    )
    st.plotly_chart(build_association_figure(reports["association"]))

    st.markdown("---")
    st.subheader("💡 Key findings")
    findings = compute_key_findings(reports)
    for finding in findings:
        st.markdown(f"- {finding}")

    st.markdown("---")
    st.subheader("⚠️ Limitations")
    for limitation in LIMITATIONS:
        st.markdown(f"- {limitation}")

    st.info(
        "Values on this page come from the saved Module 2 reports in "
        "`outputs/reports/` (generated by `src/population_analysis.py`). "
        "No raw dataset is loaded here."
    )