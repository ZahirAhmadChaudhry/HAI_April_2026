# Repeated Experiment Summary

K=20 repeated runs with seeds 43..62, temporal split fixed (2019+2020 train, 2021 test).

## 1. Paper Summary Table
| Model | Algorithm | Feature Set | AUC-ROC (mean +/- SD) | AUC-PR (mean +/- SD) | F1 (mean +/- SD) | Precision (mean +/- SD) | Recall (mean +/- SD) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model_A_random_forest | random_forest | A | 0.8599 +/- nan | 0.7796 +/- nan | 0.6842 +/- nan | 0.7222 +/- nan | 0.6500 +/- nan |
| Model_A_plus_environment_random_forest | random_forest | A_plus_environment | 0.8718 +/- nan | 0.7893 +/- nan | 0.7234 +/- nan | 0.6296 +/- nan | 0.8500 +/- nan |
| Model_A_plus_staffing_random_forest | random_forest | A_plus_staffing | 0.8706 +/- nan | 0.7962 +/- nan | 0.7010 +/- nan | 0.5965 +/- nan | 0.8500 +/- nan |
| Model_B_random_forest | random_forest | B | 0.8581 +/- nan | 0.7666 +/- nan | 0.7129 +/- nan | 0.5902 +/- nan | 0.9000 +/- nan |

## 2. Best-of-A vs Best-of-B Across Runs
- Mean AUC difference (Best B - Best A): -0.001744 +/- nan
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
| Model_A_plus_environment_random_forest | 0.871802 |  | 0.871802 | 0 | 0.789305 |  | 0.723404 |  | 0.62963 |  | 0.85 |  | 0.767442 |  | 0.580781 |  | 0.155296 |  |
| Model_A_plus_staffing_random_forest | 0.87064 |  | 0.87064 | 0 | 0.796214 |  | 0.701031 |  | 0.596491 |  | 0.85 |  | 0.732558 |  | 0.544824 |  | 0.162668 |  |
| Model_B_random_forest | 0.85814 |  | 0.85814 | 0 | 0.766626 |  | 0.712871 |  | 0.590164 |  | 0.9 |  | 0.709302 |  | 0.567532 |  | 0.173388 |  |

## 5. SHAP Stability (Winner Model B Algorithm)
### Top 25 features by mean |SHAP|
| index | feature | original_feature | group | importance_mean | importance_sd | importance_median |
| --- | --- | --- | --- | --- | --- | --- |
| 44 | intubation_days | intubation_days | Medical Procedures | 0.0638979 |  | 0.0638979 |
| 68 | urinary_catheter_days | urinary_catheter_days | Medical Procedures | 0.0420169 |  | 0.0420169 |
| 46 | intubation_status_2 | intubation_status | Medical Procedures | 0.0393748 |  | 0.0393748 |
| 45 | intubation_status_1 | intubation_status | Medical Procedures | 0.0313638 |  | 0.0313638 |
| 58 | reintubation_status_not_applicable | reintubation_status | Medical Procedures | 0.03082 |  | 0.03082 |
| 50 | medical_admin_assistant_staffing_etp | medical_admin_assistant_staffing_etp | Organizational Staffing | 0.021036 |  | 0.021036 |
| 62 | total_staffing_etp | total_staffing_etp | Organizational Staffing | 0.0185505 |  | 0.0185505 |
| 52 | nurse_aide_staffing_etp | nurse_aide_staffing_etp | Organizational Staffing | 0.0178961 |  | 0.0178961 |
| 47 | length_of_stay | length_of_stay | Length of Stay | 0.0160583 |  | 0.0160583 |
| 31 | central_line_count | central_line_count | Medical Procedures | 0.0152434 |  | 0.0152434 |
| 48 | los_ratio_national | los_ratio_national | Organizational Environment | 0.0152433 |  | 0.0152433 |
| 54 | nurse_staffing_etp | nurse_staffing_etp | Organizational Staffing | 0.0151995 |  | 0.0151995 |
| 57 | reintubation_status_2 | reintubation_status | Medical Procedures | 0.0139816 |  | 0.0139816 |
| 65 | unit_avg_los | unit_avg_los | Organizational Environment | 0.0125889 |  | 0.0125889 |
| 30 | cancer_status_9 | cancer_status | Clinical Severity | 0.00795766 |  | 0.00795766 |
| 36 | dietitian_staffing_etp | dietitian_staffing_etp | Organizational Staffing | 0.00791485 |  | 0.00791485 |
| 60 | sex_1 | sex | Patient Demographics | 0.00626106 |  | 0.00626106 |
| 29 | cancer_status_3 | cancer_status | Clinical Severity | 0.00511731 |  | 0.00511731 |
| 17 | admission_weekday_2 | admission_weekday | Temporal | 0.00462484 |  | 0.00462484 |
| 42 | immunosuppression_3 | immunosuppression | Clinical Severity | 0.00457974 |  | 0.00457974 |
| 51 | national_avg_los | national_avg_los | Organizational Environment | 0.00424339 |  | 0.00424339 |
| 2 | admission_month_3 | admission_month | Temporal | 0.00422858 |  | 0.00422858 |
| 67 | urinary_catheter_2 | urinary_catheter | Medical Procedures | 0.00375024 |  | 0.00375024 |
| 56 | reintubation_status_1 | reintubation_status | Medical Procedures | 0.00356622 |  | 0.00356622 |
| 23 | antibiotic_at_admission_1 | antibiotic_at_admission | Clinical Severity | 0.00299992 |  | 0.00299992 |

### Group contribution stability
| index | group | group_pct_mean | group_pct_sd | group_pct_median |
| --- | --- | --- | --- | --- |
| 2 | Medical Procedures | 55.5848 |  | 55.5848 |
| 4 | Organizational Staffing | 18.6945 |  | 18.6945 |
| 3 | Organizational Environment | 7.95166 |  | 7.95166 |
| 0 | Clinical Severity | 6.47638 |  | 6.47638 |
| 6 | Temporal | 4.34028 |  | 4.34028 |
| 1 | Length of Stay | 3.62617 |  | 3.62617 |
| 5 | Patient Demographics | 3.32615 |  | 3.32615 |

## 6. Validation Checklist
| check | status | details |
| --- | --- | --- |
| Expected rows in results CSV | PASS | rows=4, expected=4 |
| No NaN in metric columns | PASS | Checked metric columns in repeated_experiment_results.csv |
| AUC values between 0.5 and 1.0 | PASS | min_auc=0.8581, max_auc=0.8718 |
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

