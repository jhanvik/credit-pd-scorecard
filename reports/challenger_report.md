# Challenger report (LightGBM vs logistic scorecard)

Tuning: 20 random draws from a small grid, 5-fold stratified CV inside the training split, scored by AUC. Validation is used only to compare models. The test split is untouched.

## Best hyperparameters
```
{
 "subsample": 0.7,
 "reg_lambda": 10.0,
 "num_leaves": 15,
 "n_estimators": 200,
 "min_child_samples": 50,
 "learning_rate": 0.03,
 "colsample_bytree": 1.0
}
```
Cross-validated AUC (train folds): 0.7862

## Top 5 candidates by CV AUC
| rank_test_score | mean_test_score | std_test_score | param_num_leaves | param_learning_rate | param_n_estimators | param_min_child_samples | param_subsample | param_colsample_bytree | param_reg_lambda |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.7862 | 0.0074 | 15 | 0.0300 | 200 | 50 | 0.7000 | 1.0000 | 10.0000 |
| 2 | 0.7855 | 0.0068 | 7 | 0.0500 | 100 | 100 | 1.0000 | 0.7000 | 10.0000 |
| 3 | 0.7848 | 0.0076 | 7 | 0.0300 | 400 | 100 | 1.0000 | 0.7000 | 0.0000 |
| 4 | 0.7842 | 0.0064 | 7 | 0.0300 | 100 | 100 | 0.7000 | 0.7000 | 10.0000 |
| 5 | 0.7838 | 0.0076 | 31 | 0.0300 | 100 | 50 | 1.0000 | 1.0000 | 10.0000 |

## Comparison
| model | split | auc | gini | ks | brier |
|---|---|---|---|---|---|
| scorecard (logistic, WoE) | validation | 0.7648 | 0.5296 | 0.3906 | 0.1398 |
| LightGBM challenger | validation | 0.7798 | 0.5596 | 0.4175 | 0.1369 |
| LightGBM challenger | train | 0.8231 | 0.6463 | 0.4914 | 0.1253 |

Validation AUC difference (LightGBM minus scorecard): +0.0150, 95% paired-bootstrap interval +0.0085 to +0.0217. The interval excludes zero: the challenger is better on validation by a statistically visible margin.

## Challenger calibration by risk bucket (validation)
| bucket | customers | avg_predicted_pd | observed_default_rate |
|---|---|---|---|
| 1 | 600 | 0.0420 | 0.0267 |
| 2 | 600 | 0.0624 | 0.0600 |
| 3 | 600 | 0.0851 | 0.1133 |
| 4 | 600 | 0.1099 | 0.1050 |
| 5 | 600 | 0.1358 | 0.1617 |
| 6 | 600 | 0.1625 | 0.1817 |
| 7 | 600 | 0.1956 | 0.1933 |
| 8 | 600 | 0.2623 | 0.2550 |
| 9 | 600 | 0.4093 | 0.4500 |
| 10 | 600 | 0.7063 | 0.6650 |

## Top 10 features by gain (global importance only; not an explanation of any one decision, SHAP comes later)
| feature | share_of_gain |
|---|---|
| delay_sum_recent3 | 0.333 |
| PAY_1 | 0.143 |
| max_delay | 0.132 |
| n_delay_months | 0.033 |
| PAY_AMT1 | 0.027 |
| PAY_AMT3 | 0.026 |
| LIMIT_BAL | 0.025 |
| PAY_AMT2 | 0.024 |
| pay_ratio_avg | 0.023 |
| util_max | 0.023 |

Results can differ in the last decimals across machines and library versions even with fixed seeds.
