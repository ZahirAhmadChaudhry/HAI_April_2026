# Repeated Experiment Summary

K=20 repeated runs with seeds 43..62, temporal split fixed (2019+2020 train, 2021 test).

## 1. Paper Summary Table
| Model | Algorithm | Feature Set | AUC-ROC (mean +/- SD) | AUC-PR (mean +/- SD) | F1 (mean +/- SD) | Precision (mean +/- SD) | Recall (mean +/- SD) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Model_A_random_forest | random_forest | A | 0.8586 +/- 0.0060 | 0.7740 +/- 0.0134 | 0.6957 +/- 0.0299 | 0.7179 +/- 0.0334 | 0.6800 +/- 0.0672 |
| Model_A_xgboost | xgboost | A | 0.8354 +/- 0.0201 | 0.7350 +/- 0.0357 | 0.6464 +/- 0.0370 | 0.5682 +/- 0.0621 | 0.7775 +/- 0.1287 |
| Model_A_catboost | catboost | A | 0.8342 +/- 0.0222 | 0.7279 +/- 0.0434 | 0.6516 +/- 0.0619 | 0.7218 +/- 0.0485 | 0.6050 +/- 0.1160 |
| Model_A_lightgbm | lightgbm | A | 0.8335 +/- 0.0191 | 0.7334 +/- 0.0294 | 0.6597 +/- 0.0477 | 0.6839 +/- 0.0354 | 0.6412 +/- 0.0771 |
| Model_B_random_forest | random_forest | B | 0.8599 +/- 0.0076 | 0.7696 +/- 0.0138 | 0.7063 +/- 0.0267 | 0.6148 +/- 0.0214 | 0.8362 +/- 0.0750 |
| Model_B_catboost | catboost | B | 0.8251 +/- 0.0232 | 0.7244 +/- 0.0270 | 0.6248 +/- 0.0615 | 0.6916 +/- 0.0521 | 0.5825 +/- 0.1164 |
| Model_B_lightgbm | lightgbm | B | 0.8080 +/- 0.0297 | 0.7150 +/- 0.0288 | 0.6299 +/- 0.0555 | 0.5978 +/- 0.0384 | 0.6700 +/- 0.0927 |
| Model_B_xgboost | xgboost | B | 0.7911 +/- 0.0355 | 0.6999 +/- 0.0348 | 0.5951 +/- 0.0472 | 0.5138 +/- 0.0431 | 0.7200 +/- 0.1123 |

## 2. Best-of-A vs Best-of-B Across Runs
- Mean AUC difference (Best B - Best A): -0.000603 +/- 0.006846
- Run counts: B > A: 9, B = A: 0, B < A: 11
- Paired t-test (diff vs 0): t=-0.394066, p=0.69792
- Wilcoxon signed-rank test: statistic=93.000000, p=0.654131

## 3. Algorithm Stability (Winner Counts)
| feature_set | algorithm | wins |
| --- | --- | --- |
| A | random_forest | 13 |
| A | xgboost | 4 |
| A | catboost | 3 |
| B | random_forest | 19 |
| B | catboost | 1 |

## 4. Per-Model Aggregates (mean, sd, median, IQR)
| model_id | auc_roc_mean | auc_roc_sd | auc_roc_median | auc_roc_iqr | auc_pr_mean | auc_pr_sd | f1_mean | f1_sd | precision_mean | precision_sd | recall_mean | recall_sd | specificity_mean | specificity_sd | mcc_mean | mcc_sd | brier_score_mean | brier_score_sd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model_A_random_forest | 0.858576 | 0.00596163 | 0.859302 | 0.00748547 | 0.773968 | 0.0134347 | 0.695716 | 0.0298878 | 0.717931 | 0.0333776 | 0.68 | 0.06718 | 0.873837 | 0.0292893 | 0.563262 | 0.0346752 | 0.137569 | 0.00296933 |
| Model_A_xgboost | 0.835436 | 0.0201337 | 0.840988 | 0.0375727 | 0.734968 | 0.0357206 | 0.646362 | 0.0370227 | 0.568178 | 0.0621202 | 0.7775 | 0.128734 | 0.709302 | 0.115604 | 0.46767 | 0.0528041 | 0.184997 | 0.0317899 |
| Model_A_catboost | 0.834201 | 0.0222366 | 0.834884 | 0.0305959 | 0.727906 | 0.0434418 | 0.651573 | 0.0619077 | 0.721786 | 0.0484696 | 0.605 | 0.11602 | 0.888953 | 0.0387449 | 0.521952 | 0.0707323 | 0.152428 | 0.0156398 |
| Model_A_lightgbm | 0.833547 | 0.0190528 | 0.832994 | 0.0378997 | 0.733445 | 0.0294115 | 0.659715 | 0.0477109 | 0.68386 | 0.0354048 | 0.64125 | 0.0770658 | 0.861628 | 0.0258361 | 0.512471 | 0.0598038 | 0.149663 | 0.0138772 |
| Model_B_random_forest | 0.859942 | 0.00757612 | 0.86061 | 0.0101017 | 0.769576 | 0.013827 | 0.706331 | 0.0266933 | 0.614839 | 0.0213539 | 0.83625 | 0.074989 | 0.754651 | 0.0357701 | 0.557057 | 0.0345155 | 0.163023 | 0.00971282 |
| Model_B_catboost | 0.825051 | 0.0232224 | 0.822965 | 0.0274709 | 0.724424 | 0.0269884 | 0.624762 | 0.0614597 | 0.691628 | 0.0520987 | 0.5825 | 0.116444 | 0.875581 | 0.0431958 | 0.483642 | 0.0687229 | 0.155915 | 0.0143425 |
| Model_B_lightgbm | 0.807965 | 0.0297126 | 0.81766 | 0.0470567 | 0.715001 | 0.0288222 | 0.629879 | 0.0554629 | 0.597783 | 0.0384018 | 0.67 | 0.0926936 | 0.790698 | 0.029944 | 0.448642 | 0.0774288 | 0.168148 | 0.0153037 |
| Model_B_xgboost | 0.791068 | 0.0354566 | 0.779942 | 0.0547602 | 0.699906 | 0.0347991 | 0.5951 | 0.0471912 | 0.513842 | 0.0430675 | 0.72 | 0.112273 | 0.677326 | 0.0878192 | 0.37707 | 0.0752646 | 0.207094 | 0.025435 |

## 5. SHAP Stability (Winner Model B Algorithm)
### Top 25 features by mean |SHAP|
| index | feature | original_feature | group | importance_mean | importance_sd | importance_median |
| --- | --- | --- | --- | --- | --- | --- |
| 44 | intubation_days | intubation_days | Medical Procedures | 0.0705153 | 0.00816145 | 0.071386 |
| 68 | urinary_catheter_days | urinary_catheter_days | Medical Procedures | 0.0502552 | 0.00891985 | 0.052536 |
| 46 | intubation_status_2 | intubation_status | Medical Procedures | 0.0283473 | 0.00542505 | 0.0277375 |
| 47 | length_of_stay | length_of_stay | Length of Stay | 0.0278854 | 0.0059431 | 0.0258563 |
| 58 | reintubation_status_not_applicable | reintubation_status | Medical Procedures | 0.0278342 | 0.00662664 | 0.027854 |
| 45 | intubation_status_1 | intubation_status | Medical Procedures | 0.0269305 | 0.00572748 | 0.026644 |
| 50 | medical_admin_assistant_staffing_etp | medical_admin_assistant_staffing_etp | Organizational Staffing | 0.0243254 | 0.00384722 | 0.0247383 |
| 62 | total_staffing_etp | total_staffing_etp | Organizational Staffing | 0.0223771 | 0.00354875 | 0.021897 |
| 31 | central_line_count | central_line_count | Medical Procedures | 0.0213297 | 0.00651616 | 0.0198389 |
| 52 | nurse_aide_staffing_etp | nurse_aide_staffing_etp | Organizational Staffing | 0.0182965 | 0.00392774 | 0.017302 |
| 54 | nurse_staffing_etp | nurse_staffing_etp | Organizational Staffing | 0.0147825 | 0.0051089 | 0.0159587 |
| 48 | los_ratio_national | los_ratio_national | Organizational Environment | 0.0136212 | 0.00382116 | 0.0139998 |
| 65 | unit_avg_los | unit_avg_los | Organizational Environment | 0.0128866 | 0.0046823 | 0.011856 |
| 57 | reintubation_status_2 | reintubation_status | Medical Procedures | 0.0126229 | 0.00556434 | 0.0124324 |
| 17 | admission_weekday_2 | admission_weekday | Temporal | 0.00821181 | 0.0050148 | 0.00626327 |
| 36 | dietitian_staffing_etp | dietitian_staffing_etp | Organizational Staffing | 0.00806987 | 0.00303521 | 0.00728507 |
| 30 | cancer_status_9 | cancer_status | Clinical Severity | 0.00804971 | 0.00234278 | 0.00842059 |
| 35 | dietitian_staffing_count | dietitian_staffing_count | Organizational Staffing | 0.00711407 | 0.00364065 | 0.00722306 |
| 2 | admission_month_3 | admission_month | Temporal | 0.00654895 | 0.00259615 | 0.00620378 |
| 29 | cancer_status_3 | cancer_status | Clinical Severity | 0.00607473 | 0.00235529 | 0.00571633 |
| 42 | immunosuppression_3 | immunosuppression | Clinical Severity | 0.00598926 | 0.00237589 | 0.00537043 |
| 26 | bed_occupancy | bed_occupancy | Organizational Environment | 0.005265 | 0.00202632 | 0.00558613 |
| 18 | admission_weekday_3 | admission_weekday | Temporal | 0.00525365 | 0.00271529 | 0.00558918 |
| 8 | admission_origin_3 | admission_origin | Patient Demographics | 0.00430438 | 0.00162119 | 0.00392443 |
| 61 | sex_2 | sex | Patient Demographics | 0.00384887 | 0.00192962 | 0.00378311 |

### Group contribution stability
| index | group | group_pct_mean | group_pct_sd | group_pct_median |
| --- | --- | --- | --- | --- |
| 2 | Medical Procedures | 50.1979 | 3.64809 | 50.5363 |
| 4 | Organizational Staffing | 19.3778 | 2.00569 | 19.0449 |
| 3 | Organizational Environment | 7.63203 | 1.37359 | 7.66975 |
| 6 | Temporal | 6.84566 | 2.15012 | 6.49941 |
| 0 | Clinical Severity | 6.63008 | 0.845939 | 6.4902 |
| 1 | Length of Stay | 5.66023 | 1.20008 | 5.10591 |
| 5 | Patient Demographics | 3.65636 | 0.609686 | 3.65596 |

## 6. Validation Checklist
| check | status | details |
| --- | --- | --- |
| Exactly 160 rows in results CSV | PASS | rows=160, expected=160 |
| No NaN in metric columns | PASS | Checked metric columns in repeated_experiment_results.csv |
| AUC values between 0.5 and 1.0 | PASS | min_auc=0.7302, max_auc=0.8762 |
| SD of AUC across runs is < 0.05 | PASS | max_model_auc_sd=0.0355 |
| SHAP group contributions sum to ~100% each run | PASS | Derived from percentage definition per run (sum of group percentages equals 100 by construction). |
| Paired A vs B comparison uses exactly 20 runs | PASS | paired_runs=20 |

## 7. Output Files
- repeated_experiment_results.csv
- repeated_experiment_summary.md
- repeated_shap_stability.csv
- figures/auc_distribution_boxplot.png
- figures/auc_difference_histogram.png
- figures/shap_stability_plot.png

