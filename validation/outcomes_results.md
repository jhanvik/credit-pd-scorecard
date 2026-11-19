# Outcomes analysis (validator script, validation split)

Data already seen by the developer; a re-examination, not a fresh test.

## Calibration summary
| measure | value |
|---|---|
| mean predicted PD | 0.2179 |
| observed default rate | 0.2212 |
| calibration intercept (ideal 0) | -0.0527 |
| calibration slope (ideal 1) | 0.9325 |
| slope 95% CI low | 0.8708 |
| slope 95% CI high | 0.9942 |


## Decile calibration
| decile | n | avg_pd | observed | gap | outside_95 |
|---|---|---|---|---|---|
| 1 | 600 | 0.0630 | 0.0550 | -0.0080 |  |
| 2 | 600 | 0.0847 | 0.0817 | -0.0031 |  |
| 3 | 600 | 0.0975 | 0.0817 | -0.0158 |  |
| 4 | 600 | 0.1110 | 0.1150 | 0.0040 |  |
| 5 | 600 | 0.1256 | 0.1517 | 0.0261 |  |
| 6 | 600 | 0.1427 | 0.1817 | 0.0389 | YES |
| 7 | 600 | 0.1743 | 0.2150 | 0.0407 | YES |
| 8 | 600 | 0.2618 | 0.2350 | -0.0268 |  |
| 9 | 600 | 0.4200 | 0.4283 | 0.0083 |  |
| 10 | 600 | 0.6980 | 0.6667 | -0.0314 |  |

Rank-ordering reversals in observed rate across deciles: 0.

## AUC by segment (500 bootstrap resamples, seed 42)
| segment | n | default_rate | auc | ci_low | ci_high |
|---|---|---|---|---|---|
| low limit | 2196 | 0.3042 | 0.7558 | 0.7325 | 0.7775 |
| mid limit | 1905 | 0.1974 | 0.7508 | 0.7255 | 0.7782 |
| high limit | 1899 | 0.1490 | 0.7327 | 0.6992 | 0.7631 |
| PAY_1 <= 0 (not late) | 4690 | 0.1446 | 0.6701 | 0.6462 | 0.6930 |
| PAY_1 >= 1 (late) | 1310 | 0.4954 | 0.7107 | 0.6819 | 0.7389 |

