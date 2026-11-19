# Sensitivity and stress tests (validator script, validation split)

Frozen scorecard cutoff 0.3. Shocks are illustrative, chosen by the validator.

## Input shocks
| scenario | mean_pd | auc | flag_rate | pct_flag_changed | mean_abs_pd_change |
|---|---|---|---|---|---|
| Baseline | 0.2179 | 0.7648 | 0.2188 | 0.0000 | 0.0000 |
| LIMIT_BAL x0.8 | 0.2265 | 0.7635 | 0.2265 | 0.0090 | 0.0090 |
| BILL_AMT1-6 x1.2 | 0.2220 | 0.7642 | 0.2237 | 0.0048 | 0.0041 |
| PAY_AMT1-6 x0.8 | 0.2233 | 0.7630 | 0.2233 | 0.0045 | 0.0054 |
| PAY_1 +1 (capped at 9) | 0.3234 | 0.7182 | 0.4152 | 0.2010 | 0.1139 |
| All PAY_x +1 (capped at 9) | 0.3254 | 0.7183 | 0.4152 | 0.2020 | 0.1128 |
| Combined: limit x0.8, bill x1.2, pay_amt x0.8 | 0.2353 | 0.7595 | 0.2350 | 0.0162 | 0.0174 |


## Missing-data behaviour (one column blanked at a time)
| blanked_column | valid_pds | mean_pd | auc |
|---|---|---|---|
| LIMIT_BAL | yes | 0.2170 | 0.7596 |
| PAY_1 | yes | 0.2104 | 0.7326 |
| BILL_AMT1 | yes | 0.2173 | 0.7648 |
| PAY_AMT1 | yes | 0.2179 | 0.7631 |

