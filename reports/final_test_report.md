# Final test-set report (one-time evaluation)

Test rows: 6,000 (defaults 1,327). Models, bins, features and cutoffs were FROZEN before this run (cutoffs chosen on validation: {"scorecard": 0.3, "LightGBM": 0.36}). Nothing was refitted or retuned. The marker file TEST_SET_USED.json records this run; changing the model in response to these numbers would invalidate them.

## Discrimination and calibration (validation vs test, same code)
| model | split | auc | gini | ks | brier | auc_ci_low | auc_ci_high |
|---|---|---|---|---|---|---|---|
| LightGBM | validation | 0.7798 | 0.5596 | 0.4175 | 0.1369 | 0.7664 | 0.7940 |
| LightGBM | TEST | 0.7919 | 0.5837 | 0.4479 | 0.1314 | 0.7784 | 0.8058 |
| scorecard | validation | 0.7648 | 0.5296 | 0.3906 | 0.1398 | 0.7506 | 0.7803 |
| scorecard | TEST | 0.7717 | 0.5433 | 0.4240 | 0.1357 | 0.7573 | 0.7867 |

Test AUC difference (LightGBM minus scorecard): +0.0202, 95% paired-bootstrap interval +0.0137 to +0.0268.

## Business impact at the frozen cutoffs (ILLUSTRATIVE assumptions)
| lgd | effect | outreach_cost | false_alarm_cost |
|---|---|---|---|
| 0.60 | 0.20 | 100.00 | 1,500.00 |

| model | split | cutoff | net_benefit | net_ci_low | net_ci_high | treated_share | precision | recall |
|---|---|---|---|---|---|---|---|---|
| LightGBM | validation | 0.360 | 2,777,213.120 | 2,346,153.180 | 3,289,460.608 | 0.167 | 0.593 | 0.447 |
| LightGBM | TEST | 0.360 | 3,714,897.880 | 3,124,424.382 | 4,323,256.939 | 0.174 | 0.614 | 0.482 |
| scorecard | validation | 0.300 | 2,740,180.000 | 2,296,331.723 | 3,279,105.342 | 0.219 | 0.522 | 0.517 |
| scorecard | TEST | 0.300 | 3,735,308.920 | 3,123,624.875 | 4,348,015.380 | 0.227 | 0.535 | 0.550 |

Test net benefit difference (LightGBM minus scorecard), NT$: -20,411, 95% interval -212,595 to 164,658.

Baselines on the test customers:
| policy | net_benefit | treated_share |
|---|---|---|
| treat nobody | 0.000 | 0.000 |
| treat everybody | 324,058.080 | 1.000 |

## Score stability, validation to test
Score PSI: 0.0018 (stable). The split is random, so this is expected to be near zero.

## Calibration by risk bucket on TEST
### scorecard
| bucket | customers | avg_predicted_pd | observed_default_rate |
|---|---|---|---|
| 1 | 600 | 0.0635 | 0.0550 |
| 2 | 600 | 0.0839 | 0.0667 |
| 3 | 600 | 0.0970 | 0.1100 |
| 4 | 600 | 0.1106 | 0.1217 |
| 5 | 600 | 0.1265 | 0.1417 |
| 6 | 600 | 0.1448 | 0.1467 |
| 7 | 600 | 0.1806 | 0.1883 |
| 8 | 600 | 0.2715 | 0.2483 |
| 9 | 600 | 0.4307 | 0.4350 |
| 10 | 600 | 0.6973 | 0.6983 |

### LightGBM
| bucket | customers | avg_predicted_pd | observed_default_rate |
|---|---|---|---|
| 1 | 600 | 0.0429 | 0.0467 |
| 2 | 600 | 0.0621 | 0.0550 |
| 3 | 600 | 0.0835 | 0.0767 |
| 4 | 600 | 0.1093 | 0.1200 |
| 5 | 600 | 0.1348 | 0.1400 |
| 6 | 600 | 0.1627 | 0.1667 |
| 7 | 600 | 0.1980 | 0.1800 |
| 8 | 600 | 0.2731 | 0.2700 |
| 9 | 600 | 0.4211 | 0.4417 |
| 10 | 600 | 0.7079 | 0.7150 |

Limitations that still apply: random split (no out-of-time test), one 2005 Taiwanese cohort, illustrative economics.
