"""
Sleep Quality Prediction Web Application
Course: Machine Learning, TY Div B, Semester I, Academic Year 2026-27
Institute: JSPM's Rajarshi Shahu College of Engineering (RSCOE), Tathawade, Pune-33
Author: Aatmman Patil

Design System: Clean, restrained academic internal analytics tool.
- Single accent: Deep Navy #1F3A5F
- Neutral background: #FFFFFF, surface: #F6F8FA, border: #E3E8EE
- Semantic badges: Poor (#B42318), Average (#B54708), Good (#067647)
- Zero emojis anywhere in the interface.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st

from ui_helpers import (
    inject_custom_css,
    load_model_artifacts,
    load_dataset,
    get_quality_tier,
    generate_factual_observations,
    build_comparison_table,
    render_distribution_chart,
    render_model_comparison_chart,
    render_feature_importance_chart,
    render_correlation_matrix
)

# Page configuration
st.set_page_config(
    page_title="Sleep Quality Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject refined CSS
inject_custom_css()

# Load model, metadata, metrics, and dataset
model, feature_info, metrics_data = load_model_artifacts()
df_dataset = load_dataset()

# Session state initialization for reset and prediction persistence
defaults = {
    "age": int(feature_info["numeric_bounds"]["Age"]["default"]),
    "gender": "Female",
    "occupation": "Doctor",
    "sleep_duration": float(feature_info["numeric_bounds"]["Sleep Duration"]["default"]),
    "activity": int(feature_info["numeric_bounds"]["Physical Activity Level"]["default"]),
    "daily_steps": int(feature_info["numeric_bounds"]["Daily Steps"]["default"]),
    "stress": int(feature_info["numeric_bounds"]["Stress Level"]["default"]),
    "bmi": "Normal",
    "heart_rate": int(feature_info["numeric_bounds"]["Heart Rate"]["default"]),
    "systolic": int(feature_info["numeric_bounds"]["Systolic_BP"]["default"]),
    "diastolic": int(feature_info["numeric_bounds"]["Diastolic_BP"]["default"]),
    "disorder": "None"
}

# Check URL query param for automated presets BEFORE widget instantiation
qp = st.query_params
preset_param = qp.get("preset")
if preset_param == "Poor Sleep Profile":
    defaults.update({
        "age": 48, "gender": "Male", "occupation": "Sales Representative",
        "sleep_duration": 5.8, "activity": 30, "stress": 8, "bmi": "Obese",
        "systolic": 140, "diastolic": 90, "heart_rate": 85, "daily_steps": 3000,
        "disorder": "Sleep Apnea"
    })
elif preset_param == "Average Sleep Profile":
    defaults.update({
        "age": 35, "gender": "Female", "occupation": "Teacher",
        "sleep_duration": 6.4, "activity": 45, "stress": 6, "bmi": "Overweight",
        "systolic": 128, "diastolic": 84, "heart_rate": 74, "daily_steps": 5500,
        "disorder": "None"
    })
elif preset_param == "Good Sleep Profile":
    defaults.update({
        "age": 29, "gender": "Female", "occupation": "Engineer",
        "sleep_duration": 8.0, "activity": 75, "stress": 3, "bmi": "Normal",
        "systolic": 118, "diastolic": 76, "heart_rate": 68, "daily_steps": 8500,
        "disorder": "None"
    })

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None

# Auto-predict if requested in query param
if qp.get("predict") == "1" and st.session_state["prediction_result"] is None:
    init_df = pd.DataFrame([{
        "Gender": st.session_state["gender"],
        "Age": st.session_state["age"],
        "Occupation": st.session_state["occupation"],
        "Sleep Duration": st.session_state["sleep_duration"],
        "Physical Activity Level": st.session_state["activity"],
        "Stress Level": st.session_state["stress"],
        "BMI Category": st.session_state["bmi"],
        "Heart Rate": st.session_state["heart_rate"],
        "Daily Steps": st.session_state["daily_steps"],
        "Sleep Disorder": st.session_state["disorder"],
        "Systolic_BP": st.session_state["systolic"],
        "Diastolic_BP": st.session_state["diastolic"]
    }])
    raw_val = model.predict(init_df)[0]
    st.session_state["prediction_result"] = {
        "score": round(float(np.clip(raw_val, 1.0, 10.0)), 1),
        "inputs": {
            "Age": st.session_state["age"],
            "Sleep Duration": st.session_state["sleep_duration"],
            "Physical Activity Level": st.session_state["activity"],
            "Stress Level": st.session_state["stress"],
            "Heart Rate": st.session_state["heart_rate"],
            "Daily Steps": st.session_state["daily_steps"],
            "Systolic_BP": st.session_state["systolic"],
            "Diastolic_BP": st.session_state["diastolic"]
        }
    }

# Header Block
st.markdown('<div class="app-title">Sleep Quality Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Supervised regression on the Sleep Health and Lifestyle dataset</div>', unsafe_allow_html=True)
st.markdown('<div class="app-meta">Aatmman Patil | JSPM\'s RSCOE, Tathawade | TY Div B | Machine Learning Mini Project | 2026-27</div>', unsafe_allow_html=True)
st.divider()

# Navigation Tabs
tab_predict, tab_models, tab_insights, tab_about = st.tabs([
    "Predict",
    "Model Performance",
    "Data Insights",
    "About"
])


# =============================================================================
# TAB 1: PREDICT
# =============================================================================
with tab_predict:
    col_input, col_result = st.columns([0.42, 0.58], gap="large")

    with col_input:
        st.markdown("**Input Profile**")

        # Demographics
        st.markdown('<div class="form-section-header">Demographics</div>', unsafe_allow_html=True)
        gender_options = feature_info["categorical_options"]["Gender"]
        st.selectbox("Gender", gender_options, key="gender")

        col_dem_1, col_dem_2 = st.columns(2)
        with col_dem_1:
            st.number_input(
                "Age (years)",
                min_value=int(feature_info["numeric_bounds"]["Age"]["min"]),
                max_value=int(feature_info["numeric_bounds"]["Age"]["max"]),
                key="age",
                step=1
            )
        with col_dem_2:
            occ_options = feature_info["categorical_options"]["Occupation"]
            st.selectbox("Occupation", occ_options, key="occupation")

        # Sleep and Activity
        st.markdown('<div class="form-section-header">Sleep and Activity</div>', unsafe_allow_html=True)
        st.slider(
            "Sleep Duration (hours)",
            min_value=float(feature_info["numeric_bounds"]["Sleep Duration"]["min"]),
            max_value=float(feature_info["numeric_bounds"]["Sleep Duration"]["max"]),
            key="sleep_duration",
            step=0.1
        )

        col_act_1, col_act_2 = st.columns(2)
        with col_act_1:
            st.number_input(
                "Physical Activity (min/day)",
                min_value=int(feature_info["numeric_bounds"]["Physical Activity Level"]["min"]),
                max_value=int(feature_info["numeric_bounds"]["Physical Activity Level"]["max"]),
                key="activity",
                step=5
            )
        with col_act_2:
            st.number_input(
                "Daily Steps (steps/day)",
                min_value=int(feature_info["numeric_bounds"]["Daily Steps"]["min"]),
                max_value=int(feature_info["numeric_bounds"]["Daily Steps"]["max"]),
                key="daily_steps",
                step=500
            )

        # Health Metrics
        st.markdown('<div class="form-section-header">Health Metrics</div>', unsafe_allow_html=True)
        st.slider(
            "Stress Level (1 - 10)",
            min_value=int(feature_info["numeric_bounds"]["Stress Level"]["min"]),
            max_value=int(feature_info["numeric_bounds"]["Stress Level"]["max"]),
            key="stress",
            step=1
        )

        col_hlth_1, col_hlth_2 = st.columns(2)
        with col_hlth_1:
            bmi_options = feature_info["categorical_options"]["BMI Category"]
            st.selectbox("BMI Category", bmi_options, key="bmi")
        with col_hlth_2:
            st.number_input(
                "Resting Heart Rate (bpm)",
                min_value=int(feature_info["numeric_bounds"]["Heart Rate"]["min"]),
                max_value=int(feature_info["numeric_bounds"]["Heart Rate"]["max"]),
                key="heart_rate",
                step=1
            )

        col_bp_1, col_bp_2 = st.columns(2)
        with col_bp_1:
            st.number_input(
                "Systolic BP (mmHg)",
                min_value=int(feature_info["numeric_bounds"]["Systolic_BP"]["min"]),
                max_value=int(feature_info["numeric_bounds"]["Systolic_BP"]["max"]),
                key="systolic",
                step=1
            )
        with col_bp_2:
            st.number_input(
                "Diastolic BP (mmHg)",
                min_value=int(feature_info["numeric_bounds"]["Diastolic_BP"]["min"]),
                max_value=int(feature_info["numeric_bounds"]["Diastolic_BP"]["max"]),
                key="diastolic",
                step=1
            )

        disorder_options = feature_info["categorical_options"]["Sleep Disorder"]
        st.selectbox("Sleep Disorder Diagnosis", disorder_options, key="disorder")

        # Action buttons
        st.write("")
        col_btn_1, col_btn_2 = st.columns([0.7, 0.3])
        with col_btn_1:
            predict_clicked = st.button("Predict Sleep Quality", type="primary", use_container_width=True)
        with col_btn_2:
            reset_clicked = st.button("Reset", type="secondary", use_container_width=True)

        if reset_clicked:
            for k, v in defaults.items():
                st.session_state[k] = v
            st.session_state["prediction_result"] = None
            st.rerun()

        if predict_clicked:
            input_df = pd.DataFrame([{
                "Gender": st.session_state["gender"],
                "Age": st.session_state["age"],
                "Occupation": st.session_state["occupation"],
                "Sleep Duration": st.session_state["sleep_duration"],
                "Physical Activity Level": st.session_state["activity"],
                "Stress Level": st.session_state["stress"],
                "BMI Category": st.session_state["bmi"],
                "Heart Rate": st.session_state["heart_rate"],
                "Daily Steps": st.session_state["daily_steps"],
                "Sleep Disorder": st.session_state["disorder"],
                "Systolic_BP": st.session_state["systolic"],
                "Diastolic_BP": st.session_state["diastolic"]
            }])
            raw_val = model.predict(input_df)[0]
            bounded_val = float(np.clip(raw_val, 1.0, 10.0))
            st.session_state["prediction_result"] = {
                "score": round(bounded_val, 1),
                "inputs": {
                    "Age": st.session_state["age"],
                    "Sleep Duration": st.session_state["sleep_duration"],
                    "Physical Activity Level": st.session_state["activity"],
                    "Stress Level": st.session_state["stress"],
                    "Heart Rate": st.session_state["heart_rate"],
                    "Daily Steps": st.session_state["daily_steps"],
                    "Systolic_BP": st.session_state["systolic"],
                    "Diastolic_BP": st.session_state["diastolic"]
                }
            }

    with col_result:
        st.markdown("**Result**")

        if st.session_state["prediction_result"] is None:
            st.markdown(
                '<div class="empty-state">Enter a profile and select Predict Sleep Quality to generate the prediction score.</div>',
                unsafe_allow_html=True
            )
        else:
            res = st.session_state["prediction_result"]
            score = res["score"]
            user_inputs = res["inputs"]

            tier_name, badge_class = get_quality_tier(score)
            test_mae = metrics_data.get("models", {}).get("Decision Tree", {}).get("test_mae", 0.02)

            st.markdown(f"""
            <div class="result-container">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: #6B7280;">Predicted Sleep Quality</span>
                    <span class="badge {badge_class}">{tier_name}</span>
                </div>
                <div class="score-display">{score:.1f} <span class="score-max">/ 10</span></div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(score / 10.0))
            st.markdown(f'<div class="context-line">Typical error of this model on unseen data: ±{test_mae:.2f} rating points.</div>', unsafe_allow_html=True)

            # Two-column comparison table: Your inputs vs dataset average
            st.markdown("**Your inputs vs dataset average**")
            comp_df = build_comparison_table(user_inputs, feature_info)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            # Factual, rule-based diagnostic observations
            st.markdown("**Profile observations**")
            observations = generate_factual_observations(user_inputs, feature_info)
            for obs in observations:
                st.markdown(f"- {obs}")


# =============================================================================
# TAB 2: MODEL PERFORMANCE
# =============================================================================
with tab_models:
    st.markdown("**Model Evaluation and Comparison**")
    st.caption("Comparison of supervised regression models on the 20% hold-out test partition (N = 75) and 5-fold cross-validation on training data (N = 299).")

    if metrics_data:
        models_data = metrics_data.get("models", {})
        table_rows = []
        for name, m in models_data.items():
            is_best = (name == metrics_data.get("best_model"))
            status = "Selected" if is_best else "Evaluated"
            table_rows.append({
                "Model": name,
                "Status": status,
                "5-Fold CV R²": f"{m['cv_r2_mean']:.4f} ± {m['cv_r2_std']:.4f}",
                "Test MAE": f"{m['test_mae']:.4f}",
                "Test RMSE": f"{m['test_rmse']:.4f}",
                "Test R²": f"{m['test_r2']:.4f}"
            })
        m_df = pd.DataFrame(table_rows)
        st.dataframe(m_df, use_container_width=True, hide_index=True)

    st.markdown(
        "R² measures the proportion of variance explained by the model (higher is better), while MAE and RMSE quantify average prediction error in rating points on a 1-10 scale (lower is better, with RMSE penalizing large deviations more heavily)."
    )

    col_chart_1, col_chart_2 = st.columns(2)
    with col_chart_1:
        if metrics_data:
            comp_fig = render_model_comparison_chart(metrics_data)
            st.pyplot(comp_fig)
            st.caption("Hold-out test performance across MAE, RMSE, and R² for all four supervised algorithms.")
    with col_chart_2:
        if os.path.exists("outputs/actual_vs_predicted.png"):
            st.image("outputs/actual_vs_predicted.png", caption="Parity plot (actual vs predicted ratings) for the selected Decision Tree model.", use_container_width=True)


# =============================================================================
# TAB 3: DATA INSIGHTS
# =============================================================================
with tab_insights:
    st.markdown("**Dataset Exploration and Key Relationships**")

    # 4 Stat Tiles
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.metric("Total Records", "374")
    with col_s2:
        st.metric("Features", "12")
    with col_s3:
        st.metric("Target Range", "1.0 - 10.0")
    with col_s4:
        st.metric("Diagnosed Disorders", "41.4%")

    st.write("")

    if df_dataset is not None:
        user_score = st.session_state["prediction_result"]["score"] if st.session_state["prediction_result"] else None

        # 1. Target distribution
        dist_fig = render_distribution_chart(df_dataset, user_prediction=user_score)
        st.pyplot(dist_fig)
        st.caption("Distribution of self-reported Quality of Sleep ratings across all 374 dataset participants. Target ratings cluster predominantly between 6.0 and 8.0.")

        st.write("")

        col_ins_1, col_ins_2 = st.columns(2)
        with col_ins_1:
            # 2. Correlation matrix
            corr_fig = render_correlation_matrix(df_dataset)
            st.pyplot(corr_fig)
            st.caption("Pairwise Pearson correlation matrix across numerical physiological and lifestyle variables.")
        with col_ins_2:
            # 3. Feature importance
            feat_fig = render_feature_importance_chart(df_dataset)
            st.pyplot(feat_fig)
            st.caption("Gini feature importance weights extracted from the trained tree models. Sleep Duration and Stress Level account for over 85% of total predictive importance.")


# =============================================================================
# TAB 4: ABOUT
# =============================================================================
with tab_about:
    st.markdown("**About This Project**")

    st.markdown("##### Methodology")
    st.markdown("""
The prediction pipeline was developed using Python and scikit-learn. The Sleep Health and Lifestyle dataset was preprocessed by decomposing composite blood pressure strings into systolic and diastolic values, imputing unrecorded sleep disorders as the explicit category 'None', and applying one-hot encoding across categorical dimensions. 

Features were trained across four supervised learning algorithms: Ordinary Least Squares Linear Regression, Decision Tree Regressor, Random Forest Regressor, and AdaBoost Regressor. AdaBoost builds a sequence of weak learners, each focusing on the samples the previous ones predicted poorly. All models were cross-validated on an 80% training partition (N = 299) using 5-fold cross-validation and evaluated on an untouched 20% hold-out test partition (N = 75).
""")

    st.markdown("##### Limitations")
    st.markdown("""
- **Sample scale:** The dataset encompasses 374 observations. While 5-fold cross-validation mitigates variance, small dataset sizes naturally constrain empirical variation.
- **Synthetic distribution characteristics:** The dataset exhibits regular clustered correlations with minimal residual scatter. The high R² values reflect this low-noise benchmark distribution rather than uncurated clinical sleep study data.
- **Unrecorded clinical variables:** Environmental factors (ambient temperature, illumination), caffeine consumption, circadian chronotype, and polysomnographic sleep stages (REM and Deep Sleep proportions) were unrecorded.
""")

    st.markdown("##### Disclaimer")
    st.markdown("For educational purposes only. Not medical advice.")
