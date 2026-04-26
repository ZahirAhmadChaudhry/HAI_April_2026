# Phase 1-3 Analysis Report

## Phase 1: Overshadowing Effect

### Top 20 Clinical-Organizational Spearman Correlations
| clinical_feature | organizational_feature | spearman_rho | abs_spearman_rho | n_pairwise_non_missing |
| --- | --- | --- | --- | --- |
| cancer_status | nurse_staffing_etp | -0.776386 | 0.776386 | 406 |
| cancer_status | medical_admin_assistant_staffing_etp | -0.768643 | 0.768643 | 406 |
| cancer_status | total_staffing_etp | -0.761115 | 0.761115 | 406 |
| cancer_status | nurse_staffing_count | -0.735023 | 0.735023 | 406 |
| cancer_status | total_staffing_count | -0.725831 | 0.725831 | 406 |
| cancer_status | national_avg_los | -0.708554 | 0.708554 | 406 |
| cancer_status | nurse_aide_staffing_etp | -0.690857 | 0.690857 | 406 |
| cancer_status | nurse_aide_staffing_count | -0.67548 | 0.67548 | 406 |
| cancer_status | dietitian_staffing_etp | 0.594564 | 0.594564 | 406 |
| cancer_status | unit_avg_los | -0.524128 | 0.524128 | 406 |
| cancer_status | los_ratio_national | -0.456828 | 0.456828 | 406 |
| weekend_admission | patient_turnover | -0.392875 | 0.392875 | 406 |
| admission_weekday | patient_turnover | -0.307745 | 0.307745 | 406 |
| intubation_days | nurse_aide_staffing_etp | 0.30359 | 0.30359 | 406 |
| intubation_days | nurse_aide_staffing_count | 0.30279 | 0.30279 | 406 |
| admission_month | los_ratio_national | -0.298713 | 0.298713 | 406 |
| intubation_days | total_staffing_count | 0.29425 | 0.29425 | 406 |
| cancer_status | bed_occupancy | -0.290695 | 0.290695 | 406 |
| intubation_days | total_staffing_etp | 0.284382 | 0.284382 | 406 |
| intubation_days | los_ratio_national | 0.283997 | 0.283997 | 406 |

### SHAP Rank Stability (Top 15 Model A vs Model B)
| position | model_a_feature | model_a_group | model_a_mean_rank | model_a_stability_index | model_b_feature | model_b_group | model_b_mean_rank | model_b_stability_index |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | intubation_days | Medical Procedures | 1 | 0 | intubation_days | Medical Procedures | 1 | 0 |
| 2 | intubation_status | Medical Procedures | 2 | 0 | intubation_status | Medical Procedures | 2 | 0 |
| 3 | urinary_catheter_days | Medical Procedures | 3 | 0 | urinary_catheter_days | Medical Procedures | 3 | 0 |
| 4 | admission_weekday | Temporal | 4 | 0 | reintubation_status | Medical Procedures | 4 | 0 |
| 5 | cancer_status | Clinical Severity | 5 | 0 | length_of_stay | Length of Stay | 5 | 0 |
| 6 | reintubation_status | Medical Procedures | 6 | 0 | admission_weekday | Temporal | 6 | 0 |
| 7 | length_of_stay | Length of Stay | 7 | 0 | total_staffing_etp | Organizational Staffing | 7 | 0 |
| 8 | admission_month | Temporal | 8 | 0 | los_ratio_national | Organizational Environment | 8 | 0 |
| 9 | central_line_count | Medical Procedures | 9 | 0 | nurse_aide_staffing_etp | Organizational Staffing | 9 | 0 |
| 10 | admission_origin | Patient Demographics | 10 | 0 | unit_avg_los | Organizational Environment | 10 | 0 |
| 11 | sex | Patient Demographics | 11 | 0 | central_line_count | Medical Procedures | 11 | 0 |
| 12 | immunosuppression | Clinical Severity | 12 | 0 | medical_admin_assistant_staffing_etp | Organizational Staffing | 12 | 0 |
| 13 | urinary_catheter | Medical Procedures | 13 | 0 | nurse_staffing_etp | Organizational Staffing | 13 | 0 |
| 14 | antibiotic_at_admission | Clinical Severity | 14 | 0 | sex | Patient Demographics | 14 | 0 |
| 15 | weekend_admission | Temporal | 15 | 0 | admission_month | Temporal | 15 | 0 |

### Delta SHAP Paired t-tests (Top 5 Clinical Features)
| feature | group | n_pairs | mean_abs_shap_model_a | mean_abs_shap_model_b | mean_delta_b_minus_a | ttest_statistic | p_value | significant_at_0_05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| intubation_days | Medical Procedures | 126 | 0.0769106 | 0.0666147 | -0.0102959 | 8.28777 | 1.53037e-13 | True |
| intubation_status | Medical Procedures | 126 | 0.0738432 | 0.0580409 | -0.0158023 | 23.9679 | 1.42512e-48 | True |
| urinary_catheter_days | Medical Procedures | 126 | 0.0563113 | 0.0536148 | -0.00269654 | 2.9093 | 0.00428888 | True |
| admission_weekday | Temporal | 126 | 0.0441998 | 0.0276185 | -0.0165813 | 14.7049 | 4.9075e-29 | True |
| cancer_status | Clinical Severity | 126 | 0.0413508 | 0.0096547 | -0.0316961 | 41.5676 | 4.83302e-75 | True |

## Phase 2: Step-wise Ablation

### Seed-level Best Metrics by Configuration
| run_id | seed | feature_set | config_name | algorithm | auc_roc | precision | recall | cv_auc_roc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 43 | A | Config 1: Clinical-Only (Groups 1-5) | random_forest | 0.859884 | 0.722222 | 0.65 | 0.875001 |
| 1 | 43 | A_plus_environment | Config 3: Clinical + Environment (Groups 1-5 + Group 6) | random_forest | 0.859302 | 0.727273 | 0.8 | 0.845389 |
| 1 | 43 | A_plus_staffing | Config 2: Clinical + Staffing (Groups 1-5 + Group 7) | random_forest | 0.868023 | 0.72093 | 0.775 | 0.842082 |
| 1 | 43 | B | Config 4: Full Integrated (Groups 1-7) | random_forest | 0.850872 | 0.674419 | 0.725 | 0.831702 |

### Wilcoxon Comparisons (Precision / Recall)
| comparison | base_config | other_config | metric | n_pairs | base_mean | other_mean | mean_delta_other_minus_base | median_delta_other_minus_base | wilcoxon_statistic | p_value | significant_at_0_05 | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Config1_vs_Config2 | A | A_plus_staffing | precision | 1 | 0.722222 | 0.72093 | -0.00129199 | -0.00129199 | 0 | 1 | False | Precision dropped (suggests more false positives). |
| Config1_vs_Config2 | A | A_plus_staffing | recall | 1 | 0.65 | 0.775 | 0.125 | 0.125 | 0 | 1 | False | Recall improved (fewer false negatives). |
| Config1_vs_Config3 | A | A_plus_environment | precision | 1 | 0.722222 | 0.727273 | 0.00505051 | 0.00505051 | 0 | 1 | False | Precision improved (suggests fewer false positives). |
| Config1_vs_Config3 | A | A_plus_environment | recall | 1 | 0.65 | 0.8 | 0.15 | 0.15 | 0 | 1 | False | Recall improved (fewer false negatives). |
| Config1_vs_Config4 | A | B | precision | 1 | 0.722222 | 0.674419 | -0.0478036 | -0.0478036 | 0 | 1 | False | Precision dropped (suggests more false positives). |
| Config1_vs_Config4 | A | B | recall | 1 | 0.65 | 0.725 | 0.075 | 0.075 | 0 | 1 | False | Recall improved (fewer false negatives). |

## Phase 3: Confidence and Cost-Asymmetry

### Confidence Segmentation (Organizational |SHAP|)
| low_conf_n | high_conf_n | low_conf_mean_org_abs_shap | high_conf_mean_org_abs_shap | low_conf_median_org_abs_shap | high_conf_median_org_abs_shap | mannwhitney_statistic | mannwhitney_p_value | low_conf_false_positive_rate_at_0_5 | high_conf_false_positive_rate_at_0_5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 60 | 66 | 0.146266 | 0.144061 | 0.154103 | 0.146425 | 2253 | 0.183159 | 0.216667 | 0.0151515 |

### Optimal Cost-sensitive Thresholds
| scenario | weight_fn | weight_fp | threshold | true_negatives | false_positives | false_negatives | true_positives | precision | recall | total_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scenario_1_FN3_FP1 | 3 | 1 | 0.44 | 64 | 22 | 5 | 35 | 0.614035 | 0.875 | 37 |
| Scenario_2_FN5_FP1 | 5 | 1 | 0.44 | 64 | 22 | 5 | 35 | 0.614035 | 0.875 | 47 |

## Output Artifacts
- phase1_clinical_org_spearman_matrix.csv
- phase1_top20_clinical_org_correlations.csv
- phase1_rank_stability_full.csv
- phase1_rank_stability_top15_comparison.csv
- phase1_delta_shap_ttests.csv
- phase2_seed_best_metrics.csv
- phase2_wilcoxon_comparisons.csv
- phase3_confidence_org_shap_analysis.csv
- phase3_threshold_cost_curve.csv
- phase3_optimal_thresholds.csv
- figures/phase1_clinical_org_spearman_heatmap.png
- figures/phase3_model_b_precision_recall_curve.png
- figures/phase3_cost_adjusted_confusions.png

