# Sleep Quality Prediction Using Supervised Machine Learning

## Academic Information
- **Course:** Machine Learning Laboratory Mini Project
- **Class & Division:** Third Year (TY) Division B, Semester I (Academic Year 2026-27)
- **Institution:** JSPM's Rajarshi Shahu College of Engineering (RSCOE), Tathawade, Pune-411033
- **Author:** Aatmman Patil (Roll No: CS3216)
- **Department:** Department of Computer Engineering

---

## Executive Summary

This project develops an end-to-end, reproducible supervised machine learning pipeline to predict subjective sleep quality ratings (continuous scale from 1.0 to 10.0) from multi-factorial physiological, lifestyle, and occupational indicators. 

Utilizing the Sleep Health and Lifestyle benchmark dataset (N = 374), four supervised regression paradigms were implemented, optimized, and systematically benchmarked:
1. **Ordinary Least Squares (OLS) Linear Regression** (Parametric baseline)
2. **Decision Tree Regressor** (Non-parametric recursive partitioner)
3. **Random Forest Regressor** (Bagging ensemble meta-estimator)
4. **AdaBoost Regressor** (Sequential boosting meta-estimator with decision tree weak learners)

Model optimization was conducted using 5-fold cross-validation and hyperparameter grid search on an 80% training partition (N = 299). Final evaluation on an untouched 20% hold-out test partition (N = 75) demonstrated that the **Decision Tree Regressor** achieved the lowest prediction error with a Test RMSE of **0.1323**, a Test MAE of **0.0233**, and an R² of **0.9884**. **Random Forest** achieved an RMSE of **0.1346** (R² = 0.9880), while **AdaBoost** secured the highest 5-fold cross-validation score across all architectures (R² = **0.9876 +/- 0.0133**) with a Test RMSE of **0.1520**.

An interactive, professional Streamlit web application deploys the winning model for real-time inference, featuring input validation, demographic delta comparison tables, and qualitative interpretation tiers.

---

## Dataset & Preprocessing

The analysis uses the Sleep Health and Lifestyle dataset (374 observations, 13 raw attributes):
- **Target Variable:** `Quality of Sleep` (integer rating scale from 1 to 10).
- **Predictors:** Age, Gender, Occupation, Sleep Duration, Physical Activity Level, Stress Level, BMI Category, Blood Pressure, Resting Heart Rate, Daily Steps, and Sleep Disorder.

### Feature Engineering & Pipeline Integrity
- **Blood Pressure Decomposition:** Composite string values (e.g., "126/83") were parsed into two physiological components: `Systolic_BP` and `Diastolic_BP`.
- **Missing Value Handling:** Missing records in `Sleep Disorder` (219 entries) represent the clinical absence of a diagnosed disorder and were explicitly imputed as `'None'`.
- **Category Harmonization:** Synonymous BMI labels (`'Normal Weight'` and `'Normal'`) were consolidated into a single `'Normal'` category.
- **Identifier Pruning:** `Person ID` was removed to prevent artificial sequence memorization.
- **Leak-Free Transformations:** Feature scaling (`StandardScaler`) and categorical encoding (`OneHotEncoder`) were encapsulated inside scikit-learn `Pipeline` and `ColumnTransformer` objects, fitted strictly on training folds.

---

## Experimental Results

All models were evaluated on the 20% hold-out test set (N = 75) with `random_state=42`.

| Supervised Model | 5-Fold CV R² | Test MAE | Test RMSE | Test R² | Selected Hyperparameters |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression** | 0.9428 +/- 0.0205 | 0.1198 | 0.2251 | 0.9664 | N/A (Standard OLS) |
| **Decision Tree (Selected)** | **0.9756 +/- 0.0231** | **0.0233** | **0.1323** | **0.9884** | `max_depth=7`, `min_samples_leaf=2`, `min_samples_split=2` |
| **Random Forest** | 0.9794 +/- 0.0141 | 0.0396 | 0.1346 | 0.9880 | `max_depth=8`, `min_samples_split=2`, `n_estimators=100` |
| **AdaBoost Regressor** | **0.9876 +/- 0.0133** | 0.0253 | 0.1520 | 0.9847 | `base_depth=4`, `learning_rate=1.0`, `loss='square'`, `n_estimators=200` |

*Selected winning model: Decision Tree Regressor, based strictly on minimum test RMSE (0.1323).*

### Key Findings
1. **Primary Drivers:** Feature importance analysis across both Random Forest and AdaBoost confirms that `Sleep Duration` (relative importance 0.573) and `Stress Level` (relative importance 0.166) account for over 70% to 80% of total predictive weight.
2. **Boosting Dynamics:** AdaBoost's staged RMSE curve demonstrates that boosting error drops sharply within the first 10 to 15 iterations and plateaus near 0.15. The discrete, low-noise structure of the target restricts further gain from additional weak learners on this dataset size.
3. **Data Characteristics:** The high R² values reflect the structured consistency and low residual noise of the benchmark dataset rather than uncurated clinical polysomnography records.

---

## Repository Structure

```
├── .streamlit/
│   └── config.toml          # Streamlit theme configuration (light academic mode)
├── data/
│   └── Sleep_health_and_lifestyle_dataset.csv  # Benchmark dataset
├── models/
│   ├── best_model.joblib    # Serialized winning model pipeline (Decision Tree)
│   └── feature_info.json    # Feature bounds and metadata for UI validation
├── outputs/
│   ├── metrics.csv          # Formatted metrics table
│   ├── metrics.json         # Raw evaluation metrics JSON
│   ├── notes.md             # Leakage audit and model performance notes
│   ├── actual_vs_predicted.png
│   ├── adaboost_feature_importance.png
│   ├── adaboost_staged_rmse.png
│   ├── app_input.png        # UI screenshot (input form)
│   ├── app_result.png       # UI screenshot (prediction results)
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── residual_plot.png
│   ├── sleep_vs_duration.png
│   ├── sleep_vs_stress.png
│   └── target_distribution.png
├── report/
│   ├── report.tex           # Complete LaTeX technical report
│   ├── results_table.tex    # Auto-generated LaTeX results table
│   ├── make_tables.py       # Script to synchronize metrics to LaTeX
│   ├── jspm_logo.jpg        # Institutional crest
│   └── rscoe_logo.png       # Departmental crest
├── src/
│   └── train.py             # Complete training, tuning, and evaluation pipeline
├── app.py                   # Streamlit web application interface
├── ui_helpers.py            # Design tokens, CSS injection, and chart renderers
├── requirements.txt         # Project dependencies
└── README.md
```

---

## Installation & Execution

### 1. Environment Setup
Clone the repository and install dependencies inside a Python virtual environment:
```bash
git clone https://github.com/aatmman/ml_mini_project.git
cd ml_mini_project

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train Models and Generate Artifacts
To execute preprocessing, run 5-fold cross-validation grid searches across all four models, serialize the winning model, and generate all 200 dpi figures:
```bash
python src/train.py
```

### 3. Launch the Web Application
Start the Streamlit interface locally:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your web browser.

### 4. Regenerate Report Tables
To regenerate the LaTeX results table directly from the latest metrics output:
```bash
python report/make_tables.py
```

---

## Disclaimer

This software and predictive model are developed solely for academic and educational evaluation. The predictions generated do not constitute medical diagnoses, clinical assessments, or therapeutic advice.
