# Modeling Report

Generated from clean_hai_dataset.csv using the locked-in temporal split and model-comparison PRD.

## 1. Data Split Summary
- Train rows (2019+2020): 280
- Test rows (2021): 126
- Train infection rate: 20.36%
- Test infection rate: 31.75%

## 2. Model Comparison Table
| model_id | algorithm | feature_set | auc_roc_ci | auc_pr_ci | f1_ci | precision_ci | recall_ci | specificity_ci | mcc_ci | brier_ci |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model_A_catboost | catboost | A | 0.873 [0.784, 0.939] | 0.807 [0.686, 0.911] | 0.756 [0.632, 0.844] | 0.738 [0.595, 0.872] | 0.775 [0.629, 0.889] | 0.872 [0.800, 0.940] | 0.639 [0.479, 0.768] | 0.142 [0.119, 0.168] |
| Model_A_lightgbm | lightgbm | A | 0.860 [0.774, 0.928] | 0.741 [0.596, 0.885] | 0.741 [0.613, 0.837] | 0.732 [0.585, 0.872] | 0.750 [0.600, 0.865] | 0.872 [0.800, 0.940] | 0.618 [0.459, 0.749] | 0.134 [0.103, 0.172] |
| Model_A_random_forest | random_forest | A | 0.860 [0.773, 0.926] | 0.779 [0.651, 0.888] | 0.667 [0.531, 0.778] | 0.714 [0.559, 0.867] | 0.625 [0.468, 0.763] | 0.884 [0.810, 0.949] | 0.529 [0.358, 0.683] | 0.136 [0.107, 0.169] |
| Model_A_xgboost | xgboost | A | 0.846 [0.766, 0.914] | 0.760 [0.636, 0.869] | 0.673 [0.557, 0.774] | 0.569 [0.440, 0.698] | 0.825 [0.703, 0.927] | 0.709 [0.611, 0.798] | 0.499 [0.357, 0.639] | 0.175 [0.144, 0.210] |
| Model_B_catboost | catboost | B | 0.873 [0.789, 0.936] | 0.783 [0.649, 0.901] | 0.750 [0.633, 0.844] | 0.688 [0.551, 0.820] | 0.825 [0.694, 0.927] | 0.826 [0.744, 0.905] | 0.624 [0.472, 0.758] | 0.155 [0.132, 0.181] |
| Model_B_random_forest | random_forest | B | 0.869 [0.784, 0.933] | 0.783 [0.651, 0.891] | 0.716 [0.593, 0.809] | 0.618 [0.490, 0.750] | 0.850 [0.724, 0.947] | 0.756 [0.663, 0.846] | 0.569 [0.416, 0.702] | 0.172 [0.147, 0.197] |
| Model_B_lightgbm | lightgbm | B | 0.838 [0.742, 0.912] | 0.770 [0.645, 0.875] | 0.689 [0.564, 0.791] | 0.620 [0.481, 0.755] | 0.775 [0.634, 0.886] | 0.779 [0.691, 0.861] | 0.527 [0.362, 0.676] | 0.151 [0.119, 0.188] |
| Model_B_xgboost | xgboost | B | 0.814 [0.720, 0.886] | 0.724 [0.590, 0.837] | 0.617 [0.487, 0.721] | 0.537 [0.400, 0.667] | 0.725 [0.564, 0.851] | 0.709 [0.611, 0.805] | 0.409 [0.234, 0.562] | 0.175 [0.139, 0.214] |

## 3. Best Model Selection
- Best Model A: catboost (AUC-ROC=0.8727, AUC-PR=0.8072)
- Best Model B: catboost (AUC-ROC=0.8727, AUC-PR=0.7829)
- Bootstrap AUC difference (Model B - Model A): 0.0002 [95% CI: -0.0155, 0.0193], p=0.987
- Interpretation: Improvement statistically significant = False

## 4. SHAP Feature Group Contributions (Model B)
| Feature Group | Mean|SHAP| | % Contribution | Rank |
| --- | --- | --- | --- |
| Medical Procedures | 1.02001 | 51.7727 | 1 |
| Organizational Staffing | 0.370912 | 18.8265 | 2 |
| Patient Demographics | 0.171053 | 8.68218 | 3 |
| Temporal | 0.139087 | 7.05964 | 4 |
| Clinical Severity | 0.119793 | 6.08036 | 5 |
| Organizational Environment | 0.102599 | 5.20764 | 6 |
| Length of Stay | 0.0467123 | 2.37099 | 7 |

## 5. Top 15 Individual Feature Contributions
| Feature | Mean|SHAP| | Direction | Interpretation |
| --- | --- | --- | --- |
| intubation_days | 0.618129 | Higher value tends to increase risk | Feature linked to intubation_days contributes to predicted infection risk. |
| urinary_catheter_days | 0.155007 | Higher value tends to increase risk | Feature linked to urinary_catheter_days contributes to predicted infection risk. |
| intubation_status_1 | 0.0933871 | Higher value tends to increase risk | Feature linked to intubation_status contributes to predicted infection risk. |
| medical_admin_assistant_staffing_etp | 0.0772947 | Higher value tends to increase risk | Feature linked to medical_admin_assistant_staffing_etp contributes to predicted infection risk. |
| central_line_count | 0.0679379 | Higher value tends to increase risk | Feature linked to central_line_count contributes to predicted infection risk. |
| total_staffing_etp | 0.0643727 | Higher value tends to decrease risk | Feature linked to total_staffing_etp contributes to predicted infection risk. |
| sex_1 | 0.059915 | Higher value tends to increase risk | Feature linked to sex contributes to predicted infection risk. |
| nurse_staffing_count | 0.0563258 | Higher value tends to decrease risk | Feature linked to nurse_staffing_count contributes to predicted infection risk. |
| nurse_staffing_etp | 0.0557048 | Higher value tends to increase risk | Feature linked to nurse_staffing_etp contributes to predicted infection risk. |
| admission_origin_1 | 0.0501571 | Higher value tends to decrease risk | Feature linked to admission_origin contributes to predicted infection risk. |
| total_staffing_count | 0.0476085 | Higher value tends to increase risk | Feature linked to total_staffing_count contributes to predicted infection risk. |
| length_of_stay | 0.0467123 | Higher value tends to increase risk | Feature linked to length_of_stay contributes to predicted infection risk. |
| immunosuppression_3 | 0.0461177 | Higher value tends to increase risk | Feature linked to immunosuppression contributes to predicted infection risk. |
| los_ratio_national | 0.0453901 | Higher value tends to increase risk | Feature linked to los_ratio_national contributes to predicted infection risk. |
| reintubation_status_2 | 0.0396741 | Higher value tends to increase risk | Feature linked to reintubation_status contributes to predicted infection risk. |

## 6. Case Study Summaries
### Case 1 (Low risk, true negative)
- Test index: 93, predicted risk: 0.205, true label: 0, predicted label: 0
- Narrative: Case 1 (Low risk, true negative): predicted risk=0.205, true_label=0, predicted_label=0. Top SHAP features highlight the main contributors for this patient-level prediction.
- Top feature contributions:
| feature | feature_value | shap_value |
| --- | --- | --- |
| intubation_days | 0 | -0.785288 |
| urinary_catheter_days | 3 | -0.197036 |
| intubation_status_1 | 0 | -0.164112 |
| sex_1 | 0 | -0.118678 |
| admission_origin_1 | 1 | -0.108249 |

### Case 2 (Moderate risk, organizationally influenced)
- Test index: 87, predicted risk: 0.357, true label: 1, predicted label: 0
- Narrative: Case 2 (Moderate risk, organizationally influenced): predicted risk=0.357, true_label=1, predicted_label=0. Top SHAP features highlight the main contributors for this patient-level prediction.
- Top feature contributions:
| feature | feature_value | shap_value |
| --- | --- | --- |
| intubation_days | 0 | -0.760878 |
| intubation_status_1 | 0 | -0.140218 |
| sex_1 | 0 | -0.133125 |
| medical_admin_assistant_staffing_etp | 1 | 0.101152 |
| central_line_count | 2 | 0.0955514 |

### Case 3 (High risk, true positive)
- Test index: 18, predicted risk: 0.811, true label: 1, predicted label: 1
- Narrative: Case 3 (High risk, true positive): predicted risk=0.811, true_label=1, predicted_label=1. Top SHAP features highlight the main contributors for this patient-level prediction.
- Top feature contributions:
| feature | feature_value | shap_value |
| --- | --- | --- |
| intubation_days | 21 | 0.567038 |
| urinary_catheter_days | 21 | 0.166392 |
| admission_weekday_3 | 1 | 0.0932411 |
| los_ratio_national | 1.60246 | 0.0696804 |
| medical_admin_assistant_staffing_etp | 1 | 0.0683527 |

## 7. Key Findings
- Model B performance was compared directly against Model A on the same 2021 holdout cohort.
- Organizational features were quantified with SHAP and aggregated into group-level contribution percentages.
- Device-related procedure features and stay complexity indicators remained strong contributors to risk predictions.
- Calibration, discrimination, and patient-level explanations were generated for transparent model interpretation.
- All required artifacts were saved for paper-ready analysis and reproducibility.

## 8. Generated Artifacts
- model_comparison_results.csv
- best_model_A.joblib
- best_model_B.joblib
- shap_values_model_B.npz
- modeling_report.md
- figures/roc_comparison.png
- figures/pr_comparison.png
- figures/calibration_comparison.png
- figures/shap_summary_model_B.png
- figures/shap_feature_groups.png
- figures/shap_waterfall_case_1.png
- figures/shap_waterfall_case_2.png
- figures/shap_waterfall_case_3.png
- figures/shap_dependence_staffing.png
- figures/shap_dependence_occupancy.png
- figures/learning_curves.png
- figures/confusion_matrix.png

## 9. Validation Checklist
- [x] Train set contains only 2019 and 2020 patients
- [x] Test set contains only 2021 patients
- [x] SMOTE applied only to training data
- [x] Imputation fitted only on training data
- [x] No leakage features (icu_mortality, admission_year, bact_count, pneu_count) used as model features
- [x] Zero-variance columns dropped from feature matrix
- [x] Model A and Model B evaluated on identical test set
- [x] AUC difference assessed via bootstrap comparison
- [x] SHAP values computed on natural-distribution test data
- [x] All required figures saved as PNG
- [x] Test metrics include bootstrap 95% confidence intervals
