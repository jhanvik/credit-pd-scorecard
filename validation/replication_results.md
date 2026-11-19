# Replication results (validator script)

Train and validation data only; test split not read.

| check | result | status |
|---|---|---|
| R1 refit coefficients | max abs diff 0.00e+00 (PAY_1_woe) | PASS |
| R2 scoring vs stored PDs | max abs diff 2.22e-16 on 6,000 rows | PASS |
| R3a AUC (own rank formula vs sklearn) | 0.764796 vs 0.764796 | PASS |
| R3 auc vs reported 0.7648 | 0.7648 | PASS |
| R3 ks vs reported 0.3906 | 0.3906 | PASS |
| R3 brier vs reported 0.1398 | 0.1398 | PASS |
| R4 points at 50:1 odds | 600.0000 | PASS |
| R4 +20 points per doubling of odds | 20.0000 | PASS |
