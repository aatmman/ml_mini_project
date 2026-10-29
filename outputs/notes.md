# Leakage and Honesty Audit: Sleep Quality Prediction Study

**Course**: Machine Learning, TY Div B, Semester I, Academic Year 2026-27  
**Institute**: JSPM's Rajarshi Shahu College of Engineering (RSCOE), Tathawade, Pune-33  
**Selected Winning Model**: Decision Tree (Test RMSE: 0.1323, Test R²: 0.9884)

---

## 1. Data Leakage Assessment
- **Target Independence**: The target variable `Quality of Sleep` represents an ordinal self-reported rating on an integer scale from 1 to 10. None of the raw predictor features directly encode this exact target or share identical scale definitions.
- **High Correlation Determinants**:
  - `Sleep Duration` exhibits an exceptionally strong positive correlation with `Quality of Sleep` ($r \approx 0.88$).
  - `Stress Level` exhibits a pronounced negative correlation ($r \approx -0.90$).
  - These two physiological/lifestyle variables effectively capture over 85% of the total target variance. While clinically and logically intuitive (extended rest enhances perceived restorative quality, whereas heightened stress suppresses it), practitioners must recognize that in subjective questionnaire workflows, these variables are co-reported simultaneously. Therefore, the model functions largely as a multivariate mapping between subjective mental stress/sleep duration and overall quality rating rather than an independent predictive clinical assay.
- **Leakage Prevention During Preprocessing**:
  - All feature scaling (StandardScaler) and categorical encodings (OneHotEncoder) were fitted strictly within scikit-learn `Pipeline` objects on the training partition ($N_{train} = 299$, 80%) and transformed out-of-sample on the test partition ($N_{test} = 75$, 20%).
  - Person IDs were removed prior to modeling to prevent artificial ordinal memorization.

---

## 2. Dataset Limitations and Generalizability
- **Sample Scale**: The dataset contains exactly 374 observations. When partitioned 80/20, the test evaluation set contains 75 records. While 5-fold cross-validation mitigates variance on the training set, the small sample size naturally contracts the empirical variance.
- **Synthetic Characteristics**: The dataset exhibits unusually regular clustered correlations (e.g., highly linear relationships between stress, sleep duration, and occupational categories with very low residual scatter). Real-world polysomnography and actigraphy datasets feature substantial noise, reporting discrepancies, insomnia sub-phenotypes, and unmeasured confounders (e.g., caffeine, sleep environment, alcohol, shift work).
- **Metric Inflation Consideration**: The high $R^2$ values ($> 0.90$) observed across both cross-validation and test partitions should be evaluated with academic honesty: they reflect the synthetic uniformity and low noise ceiling of the benchmark dataset rather than flawless real-world predictive capability.

---

## 3. AdaBoost Regressor Performance Analysis
- **Empirical Metrics**: AdaBoost Regressor achieved a test RMSE of 0.1520, a test MAE of 0.0253, and a test R² of 0.9847, with a 5-fold cross-validation R² of 0.9876 $\pm$ 0.0133.
- **Comparative Dynamic**: Across all four evaluated architectures (Linear Regression, Decision Tree, Random Forest, and AdaBoost), the ensemble models exhibit competitive predictive accuracy. AdaBoost constructs an ensemble of shallow decision trees by sequentially up-weighting observations with higher prediction error under the chosen loss function.
- **Small Dataset Considerations**: On this small dataset ($N = 374$, with $N_{train} = 299$), the target variable is already dominated by two highly correlated features (`Sleep Duration` and `Stress Level`). Consequently, the staged RMSE curve reveals that boosting error drops sharply within the earliest iterations and quickly plateaus. Without strict regularization via shallow tree depth (max depth 2 to 4) and modest learning rates, boosting on small datasets risks fitting idiosyncratic noise patterns in hard samples. The comparative results demonstrate that while AdaBoost achieves strong test accuracy, the small benchmark size limits the differentiation among top tree-based algorithms.

---

## 4. Implications for Deployment
- The model serves as an effective educational demonstration of supervised regression, ensemble boosting, and multi-model benchmarking.
- In real-world biomedical applications, deployment requires continuous actigraphy sensor inputs, validation against medical-grade clinical sleep studies (Polysomnography / PSG), and validation across diverse cross-demographic cohorts.
