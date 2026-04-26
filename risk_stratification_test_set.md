# Risk Stratification on 2021 Test Set

Canonical model: final_model_B_rf.joblib
Dataset: clean_hai_dataset.csv
Test cohort definition: admission_year == 2021
Number of test patients: 126

## Risk Category Distribution

| Risk Category | Count | Percentage | Actual Infection Rate in Category |
|---|---:|---:|---:|
| Very Low | 1 | 0.79% | 0.00% |
| Low | 57 | 45.24% | 5.26% |
| Moderate | 22 | 17.46% | 18.18% |
| High | 39 | 30.95% | 66.67% |
| Very High | 7 | 5.56% | 100.00% |

## Aggregate Risk Summary

- Mean predicted risk score (all test patients): 0.483175
- Mean predicted risk score (infected, has_infection=1): 0.667450
- Mean predicted risk score (non-infected, has_infection=0): 0.397466
- Distribution pattern: Spread across categories (risk mass occupies multiple categories without a narrow two-peak concentration).

## Predicted Risk Score Distribution Statistics

- Min: 0.199095
- Max: 0.889781
- Mean: 0.483175
- Median: 0.415574
- Std: 0.200135
- Mean (infected): 0.667450
- Mean (non-infected): 0.397466

## Notes

- Preprocessing follows the canonical publication pipeline via the saved Model B preprocessor bundle (same feature set, duration rules, numeric imputation medians, and one-hot encoding behavior).
