# MODELING PRD: Two-Model Comparison with XAI Analysis

## Objective

Train and evaluate two model configurations to answer the paper's core research question: Do organizational features (staffing, workload, bed occupancy) add predictive value for HAI risk beyond clinical features alone?

The comparison between Model A (clinical only) and Model B (clinical + organizational) is the paper's primary finding. SHAP analysis on Model B is the paper's primary contribution.

## Input

- `clean_hai_dataset.csv` (406 rows × 44 columns, produced by build_final_pipeline.py)

## Outputs

```
1. model_comparison_results.csv        (all metrics for all models)
2. best_model_A.joblib                 (best clinical-only model)
3. best_model_B.joblib                 (best clinical+organizational model)
4. shap_values_model_B.npz             (SHAP values for full model)
5. modeling_report.md                   (complete results report)
6. figures/                             (all plots as PNG files)
   ├── roc_comparison.png              (ROC curves: Model A vs Model B)
   ├── pr_comparison.png               (PR curves: Model A vs Model B)
   ├── calibration_comparison.png      (calibration plots)
   ├── shap_summary_model_B.png        (SHAP beeswarm plot)
   ├── shap_feature_groups.png         (feature group contribution bar chart)
   ├── shap_waterfall_case_1.png       (low risk patient)
   ├── shap_waterfall_case_2.png       (moderate risk patient)
   ├── shap_waterfall_case_3.png       (high risk patient)
   ├── shap_dependence_staffing.png    (staffing vs SHAP value)
   ├── shap_dependence_occupancy.png   (bed occupancy vs SHAP value)
   ├── learning_curves.png             (train/val loss curves)
   └── confusion_matrix.png            (best model confusion matrix)
```

---

## LOCKED-IN DECISIONS

| Decision | Value |
|----------|-------|
| Temporal split | Train: 2019+2020 (280 patients), Test: 2021 (126 patients) |
| CV within training | Leave-one-year-out: train on 2019, validate on 2020 (and vice versa) |
| Algorithms | XGBoost, LightGBM, CatBoost, Random Forest |
| Hyperparameter tuning | Optuna, 100 trials per algorithm |
| Class balancing | SMOTE on training set only (after split, after imputation) |
| Model framing | Retrospective (complete-stay features allowed) |
| Primary metric | AUC-ROC |
| Statistical comparison | DeLong test for AUC-ROC difference |
| Confidence intervals | Bootstrap (1000 iterations) on test set |

---

## STEP 1: Load and Prepare Data

```python
df = pd.read_csv('clean_hai_dataset.csv')
```

### 1.1 Drop columns that should NOT be features

Drop these from the feature set (keep in df for reference but exclude from X):

| Column | Reason |
|--------|--------|
| has_infection | Target variable (y) |
| icu_mortality | Outcome variable, not a predictor (reverse causation risk) |
| admission_year | Used for splitting, not as a feature (would leak temporal info) |
| hospital_services_staffing_etp | Zero variance (all 0.0) |
| admin_assistant_staffing_etp | Zero variance (all 0.0) |
| hospital_services_staffing_count | Zero variance (all 0.0) |
| admin_assistant_staffing_count | Zero variance (all 0.0) |

### 1.2 Define feature sets

**Model A features (clinical only, ~20 features):**
- age
- sex
- admission_origin
- diagnostic_category
- trauma_status
- immunosuppression
- antibiotic_at_admission
- cancer_status
- severity_score_igs2
- intubation_status
- reintubation_status
- intubation_days
- urinary_catheter
- urinary_catheter_days
- central_line_count
- ecmo_status
- length_of_stay
- admission_month
- admission_weekday
- weekend_admission

**Model B features (clinical + organizational, ~32 features):**
All of Model A, plus:
- bed_occupancy
- patient_turnover
- unit_avg_los
- national_avg_los
- los_ratio_national
- nurse_staffing_etp
- nurse_aide_staffing_etp
- nurse_anesthetist_staffing_etp
- dietitian_staffing_etp
- medical_admin_assistant_staffing_etp
- total_staffing_etp
- nurse_staffing_count
- nurse_aide_staffing_count
- nurse_anesthetist_staffing_count
- dietitian_staffing_count
- medical_admin_assistant_staffing_count
- total_staffing_count

### 1.3 Temporal split

```python
train_mask = df['admission_year'].isin([2019, 2020])
test_mask = df['admission_year'] == 2021

y_train = df.loc[train_mask, 'has_infection']
y_test = df.loc[test_mask, 'has_infection']
```

Print: train size, test size, infection rate in each.

---

## STEP 2: Preprocessing Pipeline

Build a preprocessing pipeline that is FIT on training data only and TRANSFORMS both train and test. This pipeline must be identical for Model A and Model B (just applied to different feature subsets).

### 2.1 Identify column types

**Categorical columns** (to be one-hot encoded):
sex, admission_origin, diagnostic_category, trauma_status, immunosuppression, antibiotic_at_admission, cancer_status, intubation_status, reintubation_status, urinary_catheter, ecmo_status, admission_month, admission_weekday, weekend_admission

**Numeric columns** (to be imputed and optionally scaled):
age, severity_score_igs2, intubation_days, urinary_catheter_days, central_line_count, length_of_stay, bed_occupancy, patient_turnover, unit_avg_los, national_avg_los, los_ratio_national, and all staffing ETP/count columns

### 2.2 Imputation

- **Numeric columns**: Impute NaN with MEDIAN of training set
  - severity_score_igs2: 4 missing
  - intubation_days: 180 missing (these are patients who were not intubated, so NaN is meaningful. Consider imputing with 0 instead of median, since non-intubated patients have 0 intubation days)
  - urinary_catheter_days: 94 missing (same logic: non-catheterized patients should be 0)
  - length_of_stay: 75 missing (missing exit dates, impute with training median)

**IMPORTANT**: For intubation_days and urinary_catheter_days, check: are the NaN values only for patients where intubation_status=2 (not intubated) or urinary_catheter=2 (no catheter)? If yes, impute with 0, not median. This is clinically correct: no device = 0 duration.

- **Categorical columns**: NaN in reintubation_status (180 missing). These are patients who were not intubated, so reintubation is not applicable. Treat NaN as a separate category "not_applicable" during encoding.

### 2.3 Encoding

One-hot encode all categorical columns with `drop_first=False` (keep all categories for SHAP interpretability). Use the categories found in the TRAINING set. If the test set has a category not seen in training, handle gracefully (set to 0 for all dummies of that variable).

### 2.4 Scaling

Tree-based models (XGBoost, LightGBM, CatBoost, Random Forest) do NOT require feature scaling. Do NOT apply StandardScaler or MinMaxScaler. This keeps features in their original units, which is important for SHAP interpretability.

### 2.5 SMOTE

Apply SMOTE to the TRAINING set only, AFTER imputation and encoding.

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
```

Print: training set size before and after SMOTE, class distribution before and after.

The TEST set is NEVER resampled. It stays at natural distribution.

---

## STEP 3: Hyperparameter Tuning

For EACH algorithm × EACH feature set (8 combinations total):

### 3.1 Cross-validation strategy within training

Use leave-one-year-out CV:
- Fold 1: Train on 2019 data, validate on 2020 data
- Fold 2: Train on 2020 data, validate on 2019 data
- CV score = mean AUC-ROC across both folds

Note: SMOTE should be applied WITHIN each CV fold (on the fold's training portion only, not on the validation portion). Use an imblearn Pipeline for this.

### 3.2 Optuna tuning

100 trials per algorithm. Objective: maximize mean CV AUC-ROC.

**XGBoost search space:**
```python
max_depth: int [3, 10]
learning_rate: float [0.01, 0.3]
n_estimators: int [50, 500]
min_child_weight: int [1, 7]
subsample: float [0.6, 1.0]
colsample_bytree: float [0.6, 1.0]
gamma: float [0, 0.5]
scale_pos_weight: float [1, 5]  # additional class weight handling
```

**LightGBM search space:**
```python
num_leaves: int [20, 150]
learning_rate: float [0.01, 0.3]
n_estimators: int [50, 500]
min_child_samples: int [10, 50]
subsample: float [0.6, 1.0]
colsample_bytree: float [0.6, 1.0]
reg_alpha: float [0, 5.0]
reg_lambda: float [0, 5.0]
```

**CatBoost search space:**
```python
depth: int [4, 8]
learning_rate: float [0.01, 0.3]
iterations: int [50, 500]
l2_leaf_reg: float [1, 10]
```

**Random Forest search space:**
```python
n_estimators: int [50, 500]
max_depth: int [3, 15]
min_samples_split: int [2, 20]
min_samples_leaf: int [1, 10]
class_weight: categorical ['balanced', 'balanced_subsample', None]
```

### 3.3 Final training

After tuning, retrain each algorithm with its best hyperparameters on the FULL training set (2019+2020, with SMOTE applied).

---

## STEP 4: Evaluation on Test Set (2021)

For each of the 8 models (4 algorithms × 2 feature sets):

### 4.1 Metrics to compute

| Metric | Description |
|--------|-------------|
| AUC-ROC | Area under ROC curve |
| AUC-PR | Area under Precision-Recall curve |
| Accuracy | Overall accuracy |
| Precision | Positive predictive value |
| Recall (Sensitivity) | True positive rate |
| Specificity | True negative rate |
| F1 Score | Harmonic mean of precision and recall |
| MCC | Matthews Correlation Coefficient |

Use threshold = 0.5 for classification metrics (accuracy, precision, recall, F1, MCC).

### 4.2 Bootstrap confidence intervals

For each metric, compute 95% confidence intervals using bootstrap resampling (1000 iterations) on the test set.

```python
def bootstrap_metric(y_true, y_pred_proba, metric_fn, n_boot=1000):
    scores = []
    for _ in range(n_boot):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        score = metric_fn(y_true[idx], y_pred_proba[idx])
        scores.append(score)
    return np.percentile(scores, [2.5, 97.5])
```

### 4.3 DeLong test for AUC comparison

Compare the best Model A AUC vs best Model B AUC using the DeLong test.

```python
# Use scipy or a dedicated implementation
# H0: AUC_A = AUC_B
# Report: z-statistic, p-value, AUC difference with 95% CI
```

If a DeLong implementation is not readily available, use bootstrap comparison instead:
```python
# For each bootstrap sample, compute AUC_B - AUC_A
# If the 95% CI of the difference excludes 0, the difference is significant
```

### 4.4 Calibration analysis

For the best Model A and best Model B:
- Compute calibration curves (predicted probability vs observed frequency)
- Use 10 bins
- Report Brier score
- Plot calibration curves

---

## STEP 5: SHAP Analysis

### 5.1 Global SHAP (Model B only, on test set)

```python
explainer = shap.TreeExplainer(best_model_B)
shap_values = explainer.shap_values(X_test_B)
```

Generate:
1. **SHAP beeswarm plot**: All features, sorted by mean |SHAP value|
2. **SHAP bar plot**: Mean |SHAP value| per feature

### 5.2 Feature Group Analysis

Group features into these categories and compute group-level SHAP contributions:

| Group | Features |
|-------|----------|
| Patient Demographics | age, sex, admission_origin |
| Clinical Severity | diagnostic_category, trauma_status, immunosuppression, antibiotic_at_admission, cancer_status, severity_score_igs2 |
| Medical Procedures | intubation_status, reintubation_status, intubation_days, urinary_catheter, urinary_catheter_days, central_line_count, ecmo_status |
| Length of Stay | length_of_stay |
| Temporal | admission_month, admission_weekday, weekend_admission |
| Organizational Environment | bed_occupancy, patient_turnover, unit_avg_los, national_avg_los, los_ratio_national |
| Organizational Staffing | nurse_staffing_etp, nurse_aide_staffing_etp, nurse_anesthetist_staffing_etp, dietitian_staffing_etp, medical_admin_assistant_staffing_etp, total_staffing_etp, and corresponding count columns |

For each group:
```python
group_contribution = sum of mean(|SHAP|) for all features in the group
group_percentage = group_contribution / total_contribution * 100
```

Generate a horizontal bar chart showing group contributions (%).

### 5.3 SHAP Dependence Plots

Generate dependence plots for key organizational features:
1. nurse_staffing_etp vs SHAP value (colored by severity_score_igs2)
2. bed_occupancy vs SHAP value (colored by intubation_status)
3. total_staffing_etp vs SHAP value (colored by length_of_stay)
4. nurse_aide_staffing_etp vs SHAP value (colored by central_line_count)

These plots show HOW organizational features affect risk and whether the effect depends on patient clinical status (interaction effects).

### 5.4 Patient Case Studies

Select 3 patients from the test set:
1. A TRUE NEGATIVE with low predicted risk (model correctly identified low risk)
2. A TRUE POSITIVE with high predicted risk (model correctly identified high risk)
3. A case where organizational features played a significant role in the prediction

For each case:
- Generate SHAP waterfall plot (top 15 features)
- Print the patient's actual feature values alongside SHAP contributions
- Write a brief clinical narrative explaining the prediction

### 5.5 Save SHAP values

```python
np.savez('shap_values_model_B.npz',
         shap_values=shap_values,
         feature_names=X_test_B.columns.tolist(),
         expected_value=explainer.expected_value)
```

---

## STEP 6: Results Summary Report

Generate `modeling_report.md` containing:

### 6.1 Model Comparison Table

| Model | Algorithm | Feature Set | AUC-ROC [95% CI] | AUC-PR | F1 | Precision | Recall | Specificity | MCC | Brier |
|-------|-----------|-------------|-------------------|--------|----|-----------|---------|----|-----|-------|

8 rows (4 algorithms × 2 feature sets).

### 6.2 Best Model Selection

- Best Model A: which algorithm, what AUC
- Best Model B: which algorithm, what AUC
- DeLong test result: z-statistic, p-value
- AUC improvement: Model B - Model A [95% CI]
- Interpretation: is the improvement statistically significant?

### 6.3 Feature Group Contributions (from SHAP)

| Feature Group | Mean |SHAP| | % Contribution | Rank |
|---------------|-------------|----------------|------|

### 6.4 Top 15 Individual Feature Contributions

| Feature | Mean |SHAP| | Direction | Interpretation |
|---------|-------------|-----------|----------------|

### 6.5 Case Study Summaries

For each of the 3 selected patients: feature values, predicted risk, actual outcome, top contributing features with SHAP values, clinical interpretation.

### 6.6 Key Findings

Write 3-5 bullet points summarizing the main findings, suitable for the paper's Results and Discussion sections.

---

## STEP 7: Save All Artifacts

Save everything to the working directory:
- All model files (.joblib)
- All figures (.png, 150 DPI, publication quality)
- Results CSV
- SHAP values
- Complete report

---

## TECHNICAL NOTES

### Handling the leave-one-year-out CV with SMOTE

The CV loop must apply SMOTE separately within each fold:

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Inside the Optuna objective:
for train_years, val_years in [(2019, 2020), (2020, 2019)]:
    fold_train_mask = year_labels == train_years
    fold_val_mask = year_labels == val_years
    
    X_fold_train = X_train[fold_train_mask]
    y_fold_train = y_train[fold_train_mask]
    X_fold_val = X_train[fold_val_mask]
    y_fold_val = y_train[fold_val_mask]
    
    # SMOTE on fold training only
    smote = SMOTE(random_state=42)
    X_fold_resampled, y_fold_resampled = smote.fit_resample(X_fold_train, y_fold_train)
    
    # Train and evaluate on fold validation
    model.fit(X_fold_resampled, y_fold_resampled)
    val_score = roc_auc_score(y_fold_val, model.predict_proba(X_fold_val)[:, 1])
```

### Figure quality

All figures should be publication quality:
- 150 DPI minimum
- Clear axis labels with units
- Legend inside or adjacent to plot
- Font size: 12pt for axis labels, 10pt for tick labels
- Color scheme: consistent across all plots
- Save as PNG

### Reproducibility

Set random seeds everywhere:
```python
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```

---

## VALIDATION CHECKLIST

- [ ] Train set contains ONLY 2019 and 2020 patients
- [ ] Test set contains ONLY 2021 patients
- [ ] SMOTE applied only to training data
- [ ] Imputation fitted only on training data
- [ ] No data leakage: icu_mortality, admission_year, bact_count, pneu_count not in features
- [ ] Zero-variance columns dropped
- [ ] Model A and Model B evaluated on identical test set
- [ ] DeLong test or bootstrap comparison performed
- [ ] SHAP values computed on unprocessed (non-SMOTE) test data
- [ ] All figures saved as PNG
- [ ] All metrics include 95% confidence intervals
