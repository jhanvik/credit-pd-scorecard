# Stability report (PSI)

Reference = training split. Bands: <0.10 stable, 0.10-0.25 moderate, >0.25 significant (conventions).

Score PSI, train vs validation: 0.0019 (stable).

Splits are random, so near-zero PSI here is expected by construction and does NOT prove stability over time. Real monitoring needs later data (the monitoring plan in the validation report).

## Selected features, train vs validation
| feature | psi | reading |
|---|---|---|
| PAY_AMT3 | 0.0033 | stable |
| util_avg | 0.0032 | stable |
| PAY_AMT1 | 0.0026 | stable |
| pay_to_limit | 0.0023 | stable |
| PAY_AMT2 | 0.0023 | stable |
| PAY_AMT4 | 0.0019 | stable |
| LIMIT_BAL | 0.0015 | stable |
| max_delay | 0.0012 | stable |
| PAY_1 | 0.0010 | stable |
| PAY_6 | 0.0004 | stable |
| PAY_5 | 0.0002 | stable |
| PAY_4 | 0.0001 | stable |

## Illustrative stress test (synthetic shift applied to a copy of validation)
util_avg multiplied by 1.3 (capped at 1.5); PAY_1 raised by one step for customers with PAY_1 >= 0.

| feature | psi | reading |
|---|---|---|
| PAY_1 | 4.7631 | significant shift |
| util_avg | 0.2064 | moderate shift |
