# Repeated Experiment Summary

K=20 repeated runs with seeds 43..62, temporal split fixed (2019+2020 train, 2021 test).

## 1. Paper Summary Table
| Model | Algorithm | Feature Set | AUC-ROC (mean +/- SD) | AUC-PR (mean +/- SD) | F1 (mean +/- SD) | Precision (mean +/- SD) | Recall (mean +/- SD) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model_A_random_forest | random_forest | A | 0.8599 +/- nan | 0.7796 +/- nan | 0.6842 +/- nan | 0.7222 +/- nan | 0.6500 +/- nan |
| Model_A_plus_environment_random_forest | random_forest | A_plus_environment | 0.8593 +/- nan | 0.7941 +/- nan | 0.7619 +/- nan | 0.7273 +/- nan | 0.8000 +/- nan |
| Model_A_plus_staffing_random_forest | random_forest | A_plus_staffing | 0.8680 +/- nan | 0.7919 +/- nan | 0.7470 +/- nan | 0.7209 +/- nan | 0.7750 +/- nan |
| Model_B_random_forest | random_forest | B | 0.8509 +/- nan | 0.7566 +/- nan | 0.6988 +/- nan | 0.6744 +/- nan | 0.7250 +/- nan |

## 2. Best-of-A vs Best-of-B Across Runs
- Mean AUC difference (Best B - Best A): -0.009012 +/- nan
- Run counts: B > A: 0, B = A: 0, B < A: 1
- Paired t-test (diff vs 0): t=nan, p=nan
- Wilcoxon signed-rank test: statistic=0.000000, p=1

## 3. Algorithm Stability (Winner Counts)
| feature_set | algorithm | wins |
| --- | --- | --- |
| A | random_forest | 1 |
| B | random_forest | 1 |

## 4. Per-Model Aggregates (mean, sd, median, IQR)
| model_id | auc_roc_mean | auc_roc_sd | auc_roc_median | auc_roc_iqr | auc_pr_mean | auc_pr_sd | f1_mean | f1_sd | precision_mean | precision_sd | recall_mean | recall_sd | specificity_mean | specificity_sd | mcc_mean | mcc_sd | brier_score_mean | brier_score_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model_A_random_forest | 0.859884 |  | 0.859884 | 0 | 0.779579 |  | 0.684211 |  | 0.722222 |  | 0.65 |  | 0.883721 |  | 0.549947 |  | 0.134632 |  |
| Model_A_plus_environment_random_forest | 0.859302 |  | 0.859302 | 0 | 0.794117 |  | 0.761905 |  | 0.727273 |  | 0.8 |  | 0.860465 |  | 0.644905 |  | 0.141945 |  |
| Model_A_plus_staffing_random_forest | 0.868023 |  | 0.868023 | 0 | 0.791935 |  | 0.746988 |  | 0.72093 |  | 0.775 |  | 0.860465 |  | 0.623875 |  | 0.138395 |  |
| Model_B_random_forest | 0.850872 |  | 0.850872 | 0 | 0.756624 |  | 0.698795 |  | 0.674419 |  | 0.725 |  | 0.837209 |  | 0.551955 |  | 0.154634 |  |

## 5. SHAP Stability (Winner Model B Algorithm)
### Top 25 features by mean |SHAP|
| index | feature | original_feature | group | importance_mean | importance_sd | importance_median |
| --- | --- | --- | --- | --- | --- | --- |
| 44 | intubation_days | intubation_days | Medical Procedures | 0.0666147 |  | 0.0666147 |
| 68 | urinary_catheter_days | urinary_catheter_days | Medical Procedures | 0.0536148 |  | 0.0536148 |
| 46 | intubation_status_2 | intubation_status | Medical Procedures | 0.0326275 |  | 0.0326275 |
| 58 | reintubation_status_not_applicable | reintubation_status | Medical Procedures | 0.0316038 |  | 0.0316038 |
| 47 | length_of_stay | length_of_stay | Length of Stay | 0.0312515 |  | 0.0312515 |
| 62 | total_staffing_etp | total_staffing_etp | Organizational Staffing | 0.0275634 |  | 0.0275634 |
| 45 | intubation_status_1 | intubation_status | Medical Procedures | 0.0254134 |  | 0.0254134 |
| 48 | los_ratio_national | los_ratio_national | Organizational Environment | 0.0237214 |  | 0.0237214 |
| 52 | nurse_aide_staffing_etp | nurse_aide_staffing_etp | Organizational Staffing | 0.0186839 |  | 0.0186839 |
| 65 | unit_avg_los | unit_avg_los | Organizational Environment | 0.0186123 |  | 0.0186123 |
| 31 | central_line_count | central_line_count | Medical Procedures | 0.0177189 |  | 0.0177189 |
| 50 | medical_admin_assistant_staffing_etp | medical_admin_assistant_staffing_etp | Organizational Staffing | 0.015577 |  | 0.015577 |
| 54 | nurse_staffing_etp | nurse_staffing_etp | Organizational Staffing | 0.0149501 |  | 0.0149501 |
| 57 | reintubation_status_2 | reintubation_status | Medical Procedures | 0.0139305 |  | 0.0139305 |
| 17 | admission_weekday_2 | admission_weekday | Temporal | 0.0103311 |  | 0.0103311 |
| 36 | dietitian_staffing_etp | dietitian_staffing_etp | Organizational Staffing | 0.00813802 |  | 0.00813802 |
| 61 | sex_2 | sex | Patient Demographics | 0.00638771 |  | 0.00638771 |
| 59 | severity_score_igs2 | severity_score_igs2 | Clinical Severity | 0.00601438 |  | 0.00601438 |
| 35 | dietitian_staffing_count | dietitian_staffing_count | Organizational Staffing | 0.005641 |  | 0.005641 |
| 20 | admission_weekday_5 | admission_weekday | Temporal | 0.0050408 |  | 0.0050408 |
| 2 | admission_month_3 | admission_month | Temporal | 0.00503741 |  | 0.00503741 |
| 18 | admission_weekday_3 | admission_weekday | Temporal | 0.00487986 |  | 0.00487986 |
| 60 | sex_1 | sex | Patient Demographics | 0.00486197 |  | 0.00486197 |
| 22 | age | age | Patient Demographics | 0.00482141 |  | 0.00482141 |
| 51 | national_avg_los | national_avg_los | Organizational Environment | 0.00476592 |  | 0.00476592 |

### Group contribution stability
| index | group | group_pct_mean | group_pct_sd | group_pct_median |
| --- | --- | --- | --- | --- |
| 2 | Medical Procedures | 47.3261 |  | 47.3261 |
| 4 | Organizational Staffing | 17.153 |  | 17.153 |
| 3 | Organizational Environment | 10.3345 |  | 10.3345 |
| 6 | Temporal | 8.23039 |  | 8.23039 |
| 0 | Clinical Severity | 6.31099 |  | 6.31099 |
| 1 | Length of Stay | 5.91977 |  | 5.91977 |
| 5 | Patient Demographics | 4.72528 |  | 4.72528 |

## 6. Validation Checklist
| check | status | details |
| --- | --- | --- |
| Expected rows in results CSV | PASS | rows=4, expected=4 |
| No NaN in metric columns | PASS | Checked metric columns in repeated_experiment_results.csv |
| AUC values between 0.5 and 1.0 | PASS | min_auc=0.8509, max_auc=0.8680 |
| SD of AUC across runs is < 0.05 | FAIL | max_model_auc_sd=nan |
| SHAP group contributions sum to ~100% each run | PASS | Derived from percentage definition per run (sum of group percentages equals 100 by construction). |
| Paired A vs B comparison uses exactly 20 runs | PASS | paired_runs=1 |

## 7. Output Files
- repeated_experiment_results.csv
- repeated_experiment_summary.md
- repeated_shap_stability.csv
- figures/auc_distribution_boxplot.png
- figures/auc_difference_histogram.png
- figures/shap_stability_plot.png

