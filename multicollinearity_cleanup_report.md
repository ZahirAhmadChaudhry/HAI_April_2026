# Multicollinearity Cleanup Report

Computed on training set only (2019+2020), after imputation and before encoding.

## 1. Correlation Pairs with |r| > 0.80
| feature_1 | feature_2 | r | abs_r |
| --- | --- | --- | --- |
| nurse_anesthetist_staffing_etp | nurse_anesthetist_staffing_count | 1 | 1 |
| nurse_aide_staffing_etp | nurse_aide_staffing_count | 0.999413 | 0.999413 |
| total_staffing_etp | total_staffing_count | 0.997995 | 0.997995 |
| nurse_aide_staffing_etp | total_staffing_count | 0.985932 | 0.985932 |
| nurse_aide_staffing_count | total_staffing_count | 0.984996 | 0.984996 |
| nurse_aide_staffing_etp | total_staffing_etp | 0.981851 | 0.981851 |
| nurse_staffing_etp | nurse_staffing_count | 0.98135 | 0.98135 |
| total_staffing_etp | nurse_aide_staffing_count | 0.979817 | 0.979817 |
| unit_avg_los | total_staffing_etp | 0.905929 | 0.905929 |
| unit_avg_los | total_staffing_count | 0.903584 | 0.903584 |
| unit_avg_los | nurse_aide_staffing_etp | 0.894493 | 0.894493 |
| unit_avg_los | nurse_aide_staffing_count | 0.889009 | 0.889009 |
| nurse_staffing_etp | total_staffing_etp | 0.878637 | 0.878637 |
| unit_avg_los | national_avg_los | 0.87254 | 0.87254 |
| nurse_staffing_etp | total_staffing_count | 0.861274 | 0.861274 |
| total_staffing_etp | nurse_staffing_count | 0.851218 | 0.851218 |
| nurse_staffing_count | total_staffing_count | 0.841348 | 0.841348 |
| dietitian_staffing_etp | dietitian_staffing_count | 0.826528 | 0.826528 |
| national_avg_los | total_staffing_etp | 0.815193 | 0.815193 |
| national_avg_los | total_staffing_count | 0.800152 | 0.800152 |

## 2. Correlation Pairs with |r| > 0.90
| feature_1 | feature_2 | r | abs_r |
| --- | --- | --- | --- |
| nurse_anesthetist_staffing_etp | nurse_anesthetist_staffing_count | 1 | 1 |
| nurse_aide_staffing_etp | nurse_aide_staffing_count | 0.999413 | 0.999413 |
| total_staffing_etp | total_staffing_count | 0.997995 | 0.997995 |
| nurse_aide_staffing_etp | total_staffing_count | 0.985932 | 0.985932 |
| nurse_aide_staffing_count | total_staffing_count | 0.984996 | 0.984996 |
| nurse_aide_staffing_etp | total_staffing_etp | 0.981851 | 0.981851 |
| nurse_staffing_etp | nurse_staffing_count | 0.98135 | 0.98135 |
| total_staffing_etp | nurse_aide_staffing_count | 0.979817 | 0.979817 |
| unit_avg_los | total_staffing_etp | 0.905929 | 0.905929 |
| unit_avg_los | total_staffing_count | 0.903584 | 0.903584 |

## 3. Focus Correlations
| feature_1 | feature_2 | r |
| --- | --- | --- |
| nurse_staffing_etp | nurse_staffing_count | 0.98135 |
| nurse_aide_staffing_etp | nurse_aide_staffing_count | 0.999413 |
| total_staffing_etp | total_staffing_count | 0.997995 |
| nurse_staffing_etp | total_staffing_etp | 0.878637 |

## 4. VIF Table (sorted)
| feature | vif |
| --- | --- |
| total_staffing_etp | inf |
| nurse_staffing_count | inf |
| nurse_aide_staffing_count | inf |
| nurse_staffing_etp | inf |
| nurse_aide_staffing_etp | inf |
| nurse_anesthetist_staffing_etp | inf |
| dietitian_staffing_etp | inf |
| nurse_anesthetist_staffing_count | inf |
| dietitian_staffing_count | inf |
| medical_admin_assistant_staffing_count | inf |
| total_staffing_count | inf |
| medical_admin_assistant_staffing_etp | inf |
| unit_avg_los | 28.6864 |
| national_avg_los | 24.5488 |
| los_ratio_national | 6.14502 |
| urinary_catheter_days | 3.30057 |
| intubation_days | 3.02089 |
| bed_occupancy | 2.50994 |
| length_of_stay | 1.98183 |
| central_line_count | 1.6854 |
| severity_score_igs2 | 1.4677 |
| patient_turnover | 1.382 |
| age | 1.18658 |

## 5. Rule-Based Decisions
| feature | decision | reason |
| --- | --- | --- |
| age | KEEP | Retained after rules. |
| sex | KEEP | Retained after rules. |
| admission_origin | KEEP | Retained after rules. |
| diagnostic_category | KEEP | Retained after rules. |
| trauma_status | KEEP | Retained after rules. |
| immunosuppression | KEEP | Retained after rules. |
| antibiotic_at_admission | KEEP | Retained after rules. |
| cancer_status | KEEP | Retained after rules. |
| severity_score_igs2 | KEEP | Retained after rules. |
| intubation_status | KEEP | Retained after rules. |
| reintubation_status | KEEP | Retained after rules. |
| intubation_days | KEEP | Retained after rules. |
| urinary_catheter | KEEP | Retained after rules. |
| urinary_catheter_days | KEEP | Retained after rules. |
| central_line_count | KEEP | Retained after rules. |
| ecmo_status | KEEP | Retained after rules. |
| length_of_stay | KEEP | Retained after rules. |
| admission_month | KEEP | Retained after rules. |
| admission_weekday | KEEP | Retained after rules. |
| weekend_admission | KEEP | Retained after rules. |
| bed_occupancy | KEEP | Retained after rules. |
| patient_turnover | KEEP | Retained after rules. |
| unit_avg_los | KEEP | Retained after rules. |
| national_avg_los | KEEP | Retained after rules. |
| los_ratio_national | KEEP | Retained after rules. |
| nurse_staffing_etp | KEEP | Retained after rules. |
| nurse_aide_staffing_etp | KEEP | Retained after rules. |
| nurse_anesthetist_staffing_etp | KEEP | Retained after rules. |
| dietitian_staffing_etp | KEEP | Retained after rules. |
| medical_admin_assistant_staffing_etp | KEEP | Retained after rules. |
| total_staffing_etp | KEEP | Retained after rules. |
| nurse_staffing_count | DROP | Dropped by Rule 1: correlated with nurse_staffing_etp at r=0.9813 (/r/>0.85); keep ETP over count. |
| nurse_aide_staffing_count | DROP | Dropped by Rule 1: correlated with nurse_aide_staffing_etp at r=0.9994 (/r/>0.85); keep ETP over count. |
| nurse_anesthetist_staffing_count | DROP | Dropped by Rule 1: correlated with nurse_anesthetist_staffing_etp at r=1.0000 (/r/>0.85); keep ETP over count. |
| dietitian_staffing_count | KEEP | Retained after rules. |
| medical_admin_assistant_staffing_count | KEEP | Retained after rules. |
| total_staffing_count | DROP | Dropped by Rule 1: correlated with total_staffing_etp at r=0.9980 (/r/>0.85); keep ETP over count. |

## 6. Flagged-but-Kept Pairs (Rule 2)
| feature_1 | feature_2 | r | decision |
| --- | --- | --- | --- |
| nurse_aide_staffing_etp | total_staffing_etp | 0.981851 | Flagged by Rule 2 (kept both for interpretability) |

## 7. Final Cleaned Feature Sets
- Model A cleaned feature count: 20
- Model B cleaned feature count: 33
- Cleaned feature JSON: cleaned_feature_sets.json
- Correlation heatmap: D:/My_data/Internship/HAI/Data_analysis/HAI_April_2026/figures/correlation_matrix.png

