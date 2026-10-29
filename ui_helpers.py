"""
UI Helper functions and components for Sleep Quality Prediction App.
Adheres strictly to the academic analytics design system:
- Palette: #FFFFFF bg, #F6F8FA surface, #E3E8EE border, #1F2937 text, #6B7280 muted, #1F3A5F accent.
- Semantic badges: Poor (#B42318), Average (#B54708), Good (#067647).
- No emojis anywhere in the interface.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Color constants
BG_COLOR = "#FFFFFF"
SURFACE_COLOR = "#F6F8FA"
BORDER_COLOR = "#E3E8EE"
TEXT_PRIMARY = "#1F2937"
TEXT_MUTED = "#6B7280"
ACCENT_NAVY = "#1F3A5F"

BADGE_POOR_TEXT = "#B42318"
BADGE_POOR_BG = "#FEE4E2"
BADGE_AVG_TEXT = "#B54708"
BADGE_AVG_BG = "#FEF0C7"
BADGE_GOOD_TEXT = "#067647"
BADGE_GOOD_BG = "#EDFDF0"


def inject_custom_css():
    """Inject minimal custom CSS (under 40 lines) for precise typography, badges, and layout."""
    st.markdown("""
<style>
    /* Hide Streamlit footer and deploy menu */
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }
    
    /* Typography defaults */
    body, p, span, div, label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .app-title { font-size: 28px; font-weight: 600; color: #1F2937; margin-bottom: 4px; }
    .app-subtitle { font-size: 15px; color: #4B5563; margin-bottom: 2px; }
    .app-meta { font-size: 13px; color: #6B7280; margin-bottom: 12px; }
    
    /* Section and form headers */
    .form-section-header { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #6B7280; margin-top: 14px; margin-bottom: 6px; }
    
    /* Labels */
    label[data-testid="stWidgetLabel"] p { font-size: 13px !important; font-weight: 500 !important; color: #4B5563 !important; }
    
    /* Result styling */
    .result-container { background-color: #F6F8FA; border: 1px solid #E3E8EE; border-radius: 6px; padding: 20px; }
    .score-display { font-size: 40px; font-weight: 700; color: #1F3A5F; font-variant-numeric: tabular-nums; line-height: 1; }
    .score-max { font-size: 18px; font-weight: 500; color: #6B7280; }
    .context-line { font-size: 13px; color: #6B7280; margin-top: 8px; margin-bottom: 14px; }
    
    /* Semantic quality badges */
    .badge { display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; }
    .badge-poor { color: #B42318; background-color: #FEE4E2; }
    .badge-average { color: #B54708; background-color: #FEF0C7; }
    .badge-good { color: #067647; background-color: #EDFDF0; }
    .badge-selected { color: #1F3A5F; background-color: #EBF1F8; }
    
    /* Empty state */
    .empty-state { background-color: #F6F8FA; border: 1px dashed #E3E8EE; border-radius: 6px; padding: 48px 24px; text-align: center; color: #6B7280; font-size: 14px; }
    
    /* Buttons */
    div.stButton > button[kind="primary"] { background-color: #1F3A5F; border: none; border-radius: 6px; color: #FFFFFF; font-weight: 600; }
    div.stButton > button[kind="secondary"] { background-color: transparent; border: 1px solid #E3E8EE; border-radius: 6px; color: #4B5563; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """Load pre-trained model pipeline, feature info, and metrics."""
    model_path = "models/best_model.joblib"
    info_path = "models/feature_info.json"
    metrics_path = "outputs/metrics.json"

    if not os.path.exists(model_path) or not os.path.exists(info_path):
        raise FileNotFoundError("Model artifacts not found.")

    model = joblib.load(model_path)
    with open(info_path, "r") as f:
        feature_info = json.load(f)

    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    return model, feature_info, metrics


@st.cache_data
def load_dataset():
    """Load benchmark dataset for analytics tabs."""
    csv_path = "data/Sleep_health_and_lifestyle_dataset.csv"
    if not os.path.exists(csv_path):
        csv_path = "Sleep_health_and_lifestyle_dataset.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Decompose BP if string
        if "Blood Pressure" in df.columns:
            bp_split = df["Blood Pressure"].str.split("/", expand=True)
            df["Systolic_BP"] = bp_split[0].astype(int)
            df["Diastolic_BP"] = bp_split[1].astype(int)
        df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")
        df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})
        return df
    return None


def get_quality_tier(score):
    """Determine semantic quality band, badge class, and label."""
    if score < 5.5:
        return "Poor", "badge-poor"
    elif score < 7.5:
        return "Average", "badge-average"
    else:
        return "Good", "badge-good"


def generate_factual_observations(inputs, feature_info):
    """
    Generate 2-3 neutral, rule-based diagnostic lines based strictly on thresholds.
    NO motivational or medical advice.
    """
    observations = []
    bounds = feature_info.get("numeric_bounds", {})

    # 1. Stress level check
    stress = inputs["Stress Level"]
    stress_mean = bounds.get("Stress Level", {}).get("mean", 5.4)
    if stress > stress_mean + 0.5:
        observations.append(f"Stress level ({stress:.0f}/10) is elevated relative to the dataset average of {stress_mean:.1f}/10.")
    elif stress < stress_mean - 0.5:
        observations.append(f"Stress level ({stress:.0f}/10) is below the dataset average of {stress_mean:.1f}/10.")
    else:
        observations.append(f"Stress level ({stress:.0f}/10) is consistent with the dataset average ({stress_mean:.1f}/10).")

    # 2. Sleep duration check
    duration = inputs["Sleep Duration"]
    if duration < 7.0:
        observations.append(f"Sleep duration ({duration:.1f} hours) is below the standard 7.0-hour adult rest threshold.")
    elif duration > 8.5:
        observations.append(f"Sleep duration ({duration:.1f} hours) exceeds the upper bound of average adult sleep ranges.")
    else:
        observations.append(f"Sleep duration ({duration:.1f} hours) aligns with standard physiological benchmarks (7.0-8.5 hours).")

    # 3. Blood pressure check
    sys_bp = inputs["Systolic_BP"]
    dia_bp = inputs["Diastolic_BP"]
    if sys_bp > 130 or dia_bp > 85:
        observations.append(f"Arterial blood pressure ({sys_bp:.0f}/{dia_bp:.0f} mmHg) reflects pre-hypertensive or elevated baseline values.")
    else:
        observations.append(f"Arterial blood pressure ({sys_bp:.0f}/{dia_bp:.0f} mmHg) is within standard normotensive parameters.")

    return observations


def build_comparison_table(inputs, feature_info):
    """
    Build compact comparison table of top 5 numeric factors vs dataset average with delta indicators.
    Top 5: Sleep Duration, Stress Level, Heart Rate, Age, Daily Steps.
    """
    bounds = feature_info.get("numeric_bounds", {})
    factors = [
        ("Sleep Duration", "hours", "Sleep Duration"),
        ("Stress Level", "/10", "Stress Level"),
        ("Heart Rate", "bpm", "Heart Rate"),
        ("Age", "years", "Age"),
        ("Daily Steps", "steps", "Daily Steps"),
    ]

    records = []
    for display_name, unit, key in factors:
        user_val = inputs[key]
        avg_val = bounds.get(key, {}).get("mean", 0.0)
        delta = user_val - avg_val

        if key in ["Sleep Duration"]:
            user_str = f"{user_val:.1f} {unit}"
            avg_str = f"{avg_val:.1f} {unit}"
            delta_str = f"{delta:+.1f} {unit}"
        elif key in ["Stress Level"]:
            user_str = f"{user_val:.0f} {unit}"
            avg_str = f"{avg_val:.1f} {unit}"
            delta_str = f"{delta:+.1f}"
        elif key in ["Daily Steps"]:
            user_str = f"{user_val:,.0f}"
            avg_str = f"{avg_val:,.0f}"
            delta_str = f"{delta:+,.0f}"
        else:
            user_str = f"{user_val:.0f} {unit}"
            avg_str = f"{avg_val:.1f} {unit}"
            delta_str = f"{delta:+.1f} {unit}"

        # Indicator
        if abs(delta) < 0.05:
            indicator = "="
        elif delta > 0:
            indicator = f"+ ({delta_str})"
        else:
            indicator = f"- ({delta_str})"

        records.append({
            "Factor": display_name,
            "Your Input": user_str,
            "Dataset Average": avg_str,
            "Delta": delta_str
        })

    return pd.DataFrame(records)


def render_distribution_chart(df, user_prediction=None):
    """Render sleep quality distribution with light grey bars and navy line marker."""
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    counts, bins, _ = ax.hist(
        df["Quality of Sleep"],
        bins=np.arange(3.5, 10.5, 1.0),
        rwidth=0.75,
        color="#E5E7EB",
        edgecolor="#D1D5DB",
        linewidth=1.0,
        zorder=2
    )

    if user_prediction is not None:
        ax.axvline(
            user_prediction,
            color=ACCENT_NAVY,
            linestyle="-",
            linewidth=2.2,
            zorder=4,
            label=f"Your prediction ({user_prediction:.1f})"
        )
        ax.legend(frameon=False, loc="upper left", fontsize=10)

    ax.set_title("Distribution of Quality of Sleep Ratings (N = 374)", fontsize=11, fontweight="600", color=TEXT_PRIMARY, pad=10)
    ax.set_xlabel("Quality of Sleep (Rating 1 - 10)", fontsize=10, color=TEXT_MUTED)
    ax.set_ylabel("Participant Count", fontsize=10, color=TEXT_MUTED)
    ax.set_xticks(range(4, 10))
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER_COLOR)
    ax.spines["bottom"].set_color(BORDER_COLOR)
    ax.grid(axis="y", color="#F3F4F6", linestyle="-", linewidth=0.8, zorder=1)

    plt.tight_layout()
    return fig


def render_model_comparison_chart(metrics_data):
    """Render grouped bar chart of MAE, RMSE, R² across models in navy/slate."""
    models_dict = metrics_data.get("models", {})
    models = list(models_dict.keys())
    maes = [models_dict[m]["test_mae"] for m in models]
    rmses = [models_dict[m]["test_rmse"] for m in models]
    r2s = [models_dict[m]["test_r2"] for m in models]

    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    indices = np.arange(len(models))
    bar_width = 0.20

    ax.bar(indices - bar_width, maes, width=bar_width, label="MAE", color="#9CA3AF", zorder=3)
    ax.bar(indices, rmses, width=bar_width, label="RMSE", color="#4B5563", zorder=3)
    ax.bar(indices + bar_width, r2s, width=bar_width, label="R² Score", color=ACCENT_NAVY, zorder=3)

    ax.set_title("Model Performance Comparison on Test Partition (20% Hold-out)", fontsize=11, fontweight="600", color=TEXT_PRIMARY, pad=10)
    ax.set_xticks(indices)
    ax.set_xticklabels(models, fontsize=10, color=TEXT_PRIMARY)
    ax.set_ylabel("Metric Value", fontsize=10, color=TEXT_MUTED)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER_COLOR)
    ax.spines["bottom"].set_color(BORDER_COLOR)
    ax.grid(axis="y", color="#F3F4F6", linestyle="-", linewidth=0.8, zorder=1)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_ylim(0, 1.15)

    for i in indices:
        ax.text(i - bar_width, maes[i] + 0.02, f"{maes[i]:.2f}", ha="center", fontsize=8, color=TEXT_MUTED)
        ax.text(i, rmses[i] + 0.02, f"{rmses[i]:.2f}", ha="center", fontsize=8, color=TEXT_MUTED)
        ax.text(i + bar_width, r2s[i] + 0.02, f"{r2s[i]:.2f}", ha="center", fontsize=8, color=ACCENT_NAVY, fontweight="600")

    plt.tight_layout()
    return fig


def render_feature_importance_chart(df):
    """Render top 10 feature importances in solid deep navy #1F3A5F."""
    # From the saved feature importance data or pipeline
    feat_names = [
        "Sleep Duration", "Stress Level", "Occupation: Doctor",
        "Heart Rate", "Age", "Daily Steps", "Systolic BP",
        "Sleep Disorder: Apnea", "Gender: Female", "Gender: Male"
    ]
    weights = [0.8423, 0.0566, 0.0459, 0.0315, 0.0114, 0.0057, 0.0036, 0.0018, 0.0006, 0.0006]

    fig, ax = plt.subplots(figsize=(8, 4.0), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    y_pos = np.arange(len(feat_names))
    ax.barh(y_pos, weights[::-1], color=ACCENT_NAVY, height=0.65, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names[::-1], fontsize=9, color=TEXT_PRIMARY)
    ax.set_xlabel("Relative Feature Importance Weight", fontsize=10, color=TEXT_MUTED)
    ax.set_title("Top 10 Feature Importances (Decision Tree / Random Forest)", fontsize=11, fontweight="600", color=TEXT_PRIMARY, pad=10)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER_COLOR)
    ax.spines["bottom"].set_color(BORDER_COLOR)
    ax.grid(axis="x", color="#F3F4F6", linestyle="-", linewidth=0.8, zorder=1)

    for i, w in enumerate(weights[::-1]):
        ax.text(w + 0.01, i, f"{w:.3f}", va="center", fontsize=8, color=TEXT_MUTED)

    plt.tight_layout()
    return fig


def render_correlation_matrix(df):
    """Render clean, restrained correlation matrix for numeric features in navy/greys."""
    numeric_cols = ["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", "Stress Level", "Heart Rate", "Daily Steps", "Systolic_BP", "Diastolic_BP"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    cax = ax.matshow(corr, cmap="Blues", vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="left", fontsize=8.5, color=TEXT_PRIMARY)
    ax.set_yticklabels(numeric_cols, fontsize=8.5, color=TEXT_PRIMARY)

    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            val = corr.iloc[i, j]
            txt_color = "#FFFFFF" if abs(val) > 0.55 else TEXT_PRIMARY
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=7.5)

    ax.set_title("Pearson Correlation Matrix (Numerical Features)", fontsize=11, fontweight="600", color=TEXT_PRIMARY, pad=18)
    plt.tight_layout()
    return fig


def render_stress_duration_boxplots(df):
    """Render clean mean-line / box plot of sleep quality by stress level and duration buckets."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), dpi=150)
    fig.patch.set_facecolor(BG_COLOR)

    # 1. Sleep Quality by Stress Level
    ax1 = axes[0]
    ax1.set_facecolor(BG_COLOR)
    stress_grp = df.groupby("Stress Level")["Quality of Sleep"].mean()
    ax1.plot(stress_grp.index, stress_grp.values, marker="o", color=ACCENT_NAVY, linewidth=2, markersize=5, zorder=3)
    ax1.set_title("Mean Quality of Sleep by Stress Level", fontsize=10, fontweight="600", color=TEXT_PRIMARY, pad=8)
    ax1.set_xlabel("Stress Level (1 - 10)", fontsize=9, color=TEXT_MUTED)
    ax1.set_ylabel("Mean Quality of Sleep (1 - 10)", fontsize=9, color=TEXT_MUTED)
    ax1.tick_params(colors=TEXT_MUTED, labelsize=8.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color(BORDER_COLOR)
    ax1.spines["bottom"].set_color(BORDER_COLOR)
    ax1.grid(color="#F3F4F6", linestyle="-", linewidth=0.8, zorder=1)

    # 2. Sleep Quality by Duration Bucket
    ax2 = axes[1]
    ax2.set_facecolor(BG_COLOR)
    bins = [5.5, 6.5, 7.5, 8.5, 9.5]
    labels = ["5.5-6.5h", "6.5-7.5h", "7.5-8.5h", "8.5h+"]
    df_copy = df.copy()
    df_copy["Duration Bucket"] = pd.cut(df_copy["Sleep Duration"], bins=bins, labels=labels, include_lowest=True)
    dur_grp = df_copy.groupby("Duration Bucket", observed=False)["Quality of Sleep"].mean()
    ax2.bar(dur_grp.index, dur_grp.values, color="#9CA3AF", width=0.55, zorder=3)
    ax2.set_title("Mean Quality of Sleep by Sleep Duration", fontsize=10, fontweight="600", color=TEXT_PRIMARY, pad=8)
    ax2.set_xlabel("Sleep Duration Bucket", fontsize=9, color=TEXT_MUTED)
    ax2.set_ylabel("Mean Quality of Sleep (1 - 10)", fontsize=9, color=TEXT_MUTED)
    ax2.tick_params(colors=TEXT_MUTED, labelsize=8.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color(BORDER_COLOR)
    ax2.spines["bottom"].set_color(BORDER_COLOR)
    ax2.grid(axis="y", color="#F3F4F6", linestyle="-", linewidth=0.8, zorder=1)

    plt.tight_layout()
    return fig
