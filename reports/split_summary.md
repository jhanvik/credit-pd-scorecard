# Train / validation / test split

- Method: stratified on the target, seed 42, fractions {'train': 0.6, 'valid': 0.2, 'test': 0.2}
- The test set is locked: `load_split('test')` refuses unless `confirm_test=True`.
- The fingerprint (SHA-256 of the sorted IDs) lets anyone check the test set never changed.

| split | rows | defaults | default_rate | ids_sha256 (first 12) |
|---|---|---|---|---|
| train | 18,000 | 3,982 | 0.2212 | 35ba2fa78ca2 |
| valid | 6,000 | 1,327 | 0.2212 | 9ae1ab024085 |
| test | 6,000 | 1,327 | 0.2212 | 6383a8a8c1e0 |

Rows whose feature values exactly match a training row (small optimism risk): validation 10, test 13.
