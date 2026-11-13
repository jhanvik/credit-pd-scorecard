# WoE / IV report (bins fitted on the training split only)

IV bands (<0.02 not useful, <0.1 weak, <0.3 medium, <0.5 strong, above = suspicious) are conventions from credit-scoring practice, not laws.

`iv_valid` re-measures IV on the validation split using the SAME bins. A large `iv_drop` means the bins were fitted to noise.

## How to read the 'suspiciously strong' flags
The delinquency features score above IV 0.5. The leakage screen (leakage_table.md) found no single feature above AUC 0.75, and the strength comes from a small group of customers who are already 2+ months late and default at a very high rate (see the PAY_1 bin table). That is a plausible behavioral signal, so the flag is kept as a WARNING, not a removal. The residual risk is timing: the UCI text does not say how the latest status (PAY_1) relates to the outcome month. This goes in the README limitations.

## Information Value by feature
| feature | iv | strength | n_bins | iv_valid | iv_drop | valid_strength |
|---|---|---|---|---|---|---|
| delay_sum_recent3 | 0.9020 | suspiciously strong - check for leakage | 4 | 0.8345 | 0.0675 | suspiciously strong - check for leakage |
| PAY_1 | 0.9007 | suspiciously strong - check for leakage | 5 | 0.8126 | 0.0881 | suspiciously strong - check for leakage |
| n_delay_months | 0.8650 | suspiciously strong - check for leakage | 4 | 0.7934 | 0.0716 | suspiciously strong - check for leakage |
| max_delay | 0.7813 | suspiciously strong - check for leakage | 3 | 0.7050 | 0.0763 | suspiciously strong - check for leakage |
| PAY_2 | 0.5501 | suspiciously strong - check for leakage | 4 | 0.5087 | 0.0414 | suspiciously strong - check for leakage |
| PAY_3 | 0.4128 | strong | 4 | 0.3740 | 0.0388 | strong |
| PAY_4 | 0.3700 | strong | 4 | 0.3361 | 0.0339 | strong |
| PAY_5 | 0.3453 | strong | 4 | 0.3216 | 0.0237 | strong |
| PAY_6 | 0.2999 | medium | 4 | 0.2782 | 0.0217 | medium |
| pay_ratio_1 | 0.2451 | medium | 5 | 0.2494 | -0.0044 | medium |
| LIMIT_BAL | 0.1900 | medium | 6 | 0.1777 | 0.0122 | medium |
| PAY_AMT1 | 0.1755 | medium | 6 | 0.1903 | -0.0148 | medium |
| pay_ratio_avg | 0.1648 | medium | 5 | 0.1549 | 0.0099 | medium |
| PAY_AMT2 | 0.1550 | medium | 6 | 0.1595 | -0.0045 | medium |
| PAY_AMT3 | 0.1265 | medium | 6 | 0.1388 | -0.0123 | medium |
| util_avg | 0.1003 | medium | 6 | 0.0836 | 0.0167 | weak |
| PAY_AMT4 | 0.0996 | weak | 6 | 0.1220 | -0.0224 | medium |
| PAY_AMT6 | 0.0895 | weak | 6 | 0.0977 | -0.0082 | weak |
| PAY_AMT5 | 0.0859 | weak | 6 | 0.1014 | -0.0155 | medium |
| bill_trend | 0.0703 | weak | 3 | 0.0679 | 0.0024 | weak |
| util_max | 0.0677 | weak | 6 | 0.0488 | 0.0189 | weak |
| util_1 | 0.0660 | weak | 6 | 0.0441 | 0.0218 | weak |
| pay_to_limit | 0.0384 | weak | 4 | 0.0562 | -0.0178 | weak |
| BILL_AMT6 | 0.0067 | not useful | 3 | 0.0006 | 0.0061 | not useful |
| BILL_AMT1 | 0.0063 | not useful | 6 | 0.0096 | -0.0033 | not useful |
| BILL_AMT2 | 0.0056 | not useful | 5 | 0.0084 | -0.0029 | not useful |
| BILL_AMT3 | 0.0048 | not useful | 5 | 0.0073 | -0.0025 | not useful |
| BILL_AMT5 | 0.0036 | not useful | 3 | -0.0001 | 0.0036 | not useful |
| BILL_AMT4 | 0.0035 | not useful | 5 | 0.0055 | -0.0020 | not useful |

## Near-duplicate features (|corr| > 0.85 on WoE values, train)
| feature_a | feature_b | abs_corr | higher_iv |
|---|---|---|---|
| max_delay | n_delay_months | 0.939 | n_delay_months |
| BILL_AMT1 | BILL_AMT2 | 0.905 | BILL_AMT1 |
| BILL_AMT3 | BILL_AMT4 | 0.900 | BILL_AMT3 |
| BILL_AMT2 | BILL_AMT3 | 0.897 | BILL_AMT2 |
| util_1 | util_max | 0.894 | util_max |
| n_delay_months | delay_sum_recent3 | 0.888 | delay_sum_recent3 |
| util_avg | util_max | 0.870 | util_avg |

Keep the higher-IV feature of each pair, or the more business-sensible one; decide in the model phase.

## Bin tables (top 8 features by IV)
### delay_sum_recent3
| bin | count | count_share | good | bad | default_rate | woe | iv_part |
|---|---|---|---|---|---|---|---|
| < 1 | 12,884 | 0.7158 | 11,283 | 1,601 | 0.1243 | 0.6942 | 0.2796 |
| [1, 2) | 1,056 | 0.0587 | 797 | 259 | 0.2453 | -0.1355 | 0.0011 |
| [2, 6) | 2,914 | 0.1619 | 1,614 | 1,300 | 0.4461 | -1.0419 | 0.2201 |
| >= 6 | 1,146 | 0.0637 | 324 | 822 | 0.7173 | -2.1883 | 0.4011 |

### PAY_1
| bin | count | count_share | good | bad | default_rate | woe | iv_part |
|---|---|---|---|---|---|---|---|
| < -1 | 1,666 | 0.0926 | 1,465 | 201 | 0.1206 | 0.7261 | 0.0392 |
| [-1, 0) | 3,383 | 0.1879 | 2,810 | 573 | 0.1694 | 0.3313 | 0.0187 |
| [0, 1) | 8,805 | 0.4892 | 7,681 | 1,124 | 0.1277 | 0.6634 | 0.1762 |
| [1, 2) | 2,245 | 0.1247 | 1,493 | 752 | 0.3350 | -0.5726 | 0.0471 |
| >= 2 | 1,901 | 0.1056 | 569 | 1,332 | 0.7007 | -2.1082 | 0.6194 |

### n_delay_months
| bin | count | count_share | good | bad | default_rate | woe | iv_part |
|---|---|---|---|---|---|---|---|
| < 1 | 11,902 | 0.6612 | 10,541 | 1,361 | 0.1144 | 0.7885 | 0.3234 |
| [1, 2) | 2,657 | 0.1476 | 1,862 | 795 | 0.2992 | -0.4075 | 0.0272 |
| [2, 3) | 1,167 | 0.0648 | 701 | 466 | 0.3993 | -0.8502 | 0.0570 |
| >= 3 | 2,274 | 0.1263 | 914 | 1,360 | 0.5981 | -1.6554 | 0.4573 |

### max_delay
| bin | count | count_share | good | bad | default_rate | woe | iv_part |
|---|---|---|---|---|---|---|---|
| < 1 | 11,902 | 0.6612 | 10,541 | 1,361 | 0.1144 | 0.7884 | 0.3234 |
| [1, 2) | 1,019 | 0.0566 | 776 | 243 | 0.2385 | -0.0986 | 0.0006 |
| >= 2 | 5,079 | 0.2822 | 2,701 | 2,378 | 0.4682 | -1.1310 | 0.4573 |

### PAY_2
| bin | count | count_share | good | bad | default_rate | woe | iv_part |
|---|---|---|---|---|---|---|---|
| < -1 | 2,275 | 0.1264 | 1,892 | 383 | 0.1684 | 0.3381 | 0.0131 |
| [-1, 0) | 3,626 | 0.2014 | 3,039 | 587 | 0.1619 | 0.3854 | 0.0267 |
| [0, 2) | 9,426 | 0.5237 | 7,911 | 1,515 | 0.1607 | 0.3944 | 0.0725 |
| >= 2 | 2,673 | 0.1485 | 1,176 | 1,497 | 0.5600 | -1.4995 | 0.4378 |

### PAY_3
| bin | count | count_share | good | bad | default_rate | woe | iv_part |
|---|---|---|---|---|---|---|---|
| < -1 | 2,455 | 0.1364 | 2,035 | 420 | 0.1711 | 0.3189 | 0.0126 |
| [-1, 0) | 3,529 | 0.1961 | 2,986 | 543 | 0.1539 | 0.4456 | 0.0341 |
| [0, 2) | 9,490 | 0.5272 | 7,792 | 1,698 | 0.1789 | 0.2652 | 0.0343 |
| >= 2 | 2,526 | 0.1403 | 1,205 | 1,321 | 0.5230 | -1.3501 | 0.3317 |

### PAY_4
| bin | count | count_share | good | bad | default_rate | woe | iv_part |
|---|---|---|---|---|---|---|---|
| < -1 | 2,611 | 0.1451 | 2,148 | 463 | 0.1773 | 0.2755 | 0.0102 |
| [-1, 0) | 3,402 | 0.1890 | 2,863 | 539 | 0.1584 | 0.4110 | 0.0283 |
| [0, 2) | 9,861 | 0.5478 | 8,026 | 1,835 | 0.1861 | 0.2172 | 0.0243 |
| >= 2 | 2,126 | 0.1181 | 981 | 1,145 | 0.5386 | -1.4127 | 0.3073 |

### PAY_5
| bin | count | count_share | good | bad | default_rate | woe | iv_part |
|---|---|---|---|---|---|---|---|
| < -1 | 2,760 | 0.1533 | 2,258 | 502 | 0.1819 | 0.2447 | 0.0086 |
| [-1, 0) | 3,324 | 0.1847 | 2,761 | 563 | 0.1694 | 0.3312 | 0.0184 |
| [0, 2) | 10,136 | 0.5631 | 8,222 | 1,914 | 0.1888 | 0.1992 | 0.0211 |
| >= 2 | 1,780 | 0.0989 | 777 | 1,003 | 0.5635 | -1.5134 | 0.2973 |
