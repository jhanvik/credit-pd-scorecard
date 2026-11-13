# Leakage check

## 1. Decision for every column

| columns | meaning | decision | reason |
|---|---|---|---|
| ID | Row identifier | Exclude | No behavioral meaning; could leak row order. |
| LIMIT_BAL | Credit limit given to the customer | Use | A property of the account, set before the observation window (assumed from the UCI text). |
| PAY_1 .. PAY_6 | Repayment status, Sept to Apr 2005 | Use | Observed in the six months before the outcome month. Residual risk: if the Sept status were recorded after the outcome window opened it would leak; the UCI text does not say. |
| BILL_AMT1 .. 6 | Bill statement amounts, Sept to Apr | Use | Observed before the outcome month. |
| PAY_AMT1 .. 6 | Amounts paid, Sept to Apr | Use | Observed before the outcome month. |
| SEX, EDUCATION, MARRIAGE, AGE | Demographics | Exclude from the model | Fair-lending caution (verify the rules that apply to you). Kept only for the later fairness check. |
| Engineered features (10) | Ratios and delinquency counts | Use | Built from the same customer's own columns above with fixed rules; they never read the target or other customers' rows. |
| DEFAULT | default payment next month | Target | The outcome to predict. |

## 2. Automatic screen: each model input used alone (training split)
Flag threshold: AUC > 0.75 (rule of thumb). Features flagged: 0.

| feature | auc_alone | higher_value_means | rows_used | flag |
|---|---|---|---|---|
| n_delay_months | 0.731 | more default | 18,000 |  |
| max_delay | 0.721 | more default | 18,000 |  |
| delay_sum_recent3 | 0.721 | more default | 18,000 |  |
| PAY_1 | 0.694 | more default | 18,000 |  |
| PAY_2 | 0.644 | more default | 18,000 |  |
| PAY_3 | 0.631 | more default | 18,000 |  |
| LIMIT_BAL | 0.621 | less default | 18,000 |  |
| PAY_4 | 0.618 | more default | 18,000 |  |
| PAY_AMT1 | 0.611 | less default | 18,000 |  |
| pay_ratio_avg | 0.607 | less default | 17,174 |  |
| pay_ratio_1 | 0.607 | less default | 16,123 |  |
| PAY_5 | 0.605 | more default | 18,000 |  |
| PAY_AMT2 | 0.603 | less default | 18,000 |  |
| PAY_6 | 0.597 | more default | 18,000 |  |
| PAY_AMT3 | 0.594 | less default | 18,000 |  |
| PAY_AMT4 | 0.584 | less default | 18,000 |  |
| PAY_AMT6 | 0.581 | less default | 18,000 |  |
| PAY_AMT5 | 0.578 | less default | 18,000 |  |
| util_avg | 0.572 | more default | 18,000 |  |
| util_1 | 0.558 | more default | 18,000 |  |
| util_max | 0.557 | more default | 18,000 |  |
| bill_trend | 0.555 | less default | 18,000 |  |
| pay_to_limit | 0.514 | less default | 18,000 |  |
| BILL_AMT1 | 0.512 | less default | 18,000 |  |
| BILL_AMT6 | 0.506 | more default | 18,000 |  |
| BILL_AMT2 | 0.506 | less default | 18,000 |  |
| BILL_AMT3 | 0.504 | less default | 18,000 |  |
| BILL_AMT4 | 0.501 | less default | 18,000 |  |
| BILL_AMT5 | 0.500 | more default | 18,000 |  |

## 3. Demographic columns (excluded from the model), for reference

| feature | auc_alone | higher_value_means | rows_used | flag |
|---|---|---|---|---|
| SEX | 0.527 | less default | 18,000 |  |
| EDUCATION | 0.524 | more default | 18,000 |  |
| MARRIAGE | 0.515 | less default | 18,000 |  |
| AGE | 0.501 | less default | 18,000 |  |

## 4. Other checks
- Rows whose features exactly match a training row: validation 10, test 13 (negligible, but reported).
- All fitting (bin edges, IV, selection) uses the training split only.
- Assumption to verify: every input is measured before the outcome month. The UCI text lists April to September 2005 for inputs and does not name the outcome month.
