# Fairness sanity check (validation split)

Protected-type attributes are NOT model inputs; they are used only to audit outcomes. Read the caveats in `fairness.py`: base rates differ by group, no legal threshold is claimed, and the rules that apply are a compliance question to verify. Groups with fewer than 200 customers are marked too small to interpret.

## scorecard (cutoff 0.30, flags 21.9% of customers)

### SEX
| group | n | default_rate | mean_pd | calib_ratio | auc | flag_rate | tpr | fpr | precision | reliable |
|---|---|---|---|---|---|---|---|---|---|---|
| female | 3,663 | 0.213 | 0.213 | 1.004 | 0.763 | 0.209 | 0.502 | 0.129 | 0.512 | yes |
| male | 2,337 | 0.234 | 0.225 | 0.958 | 0.767 | 0.235 | 0.538 | 0.142 | 0.537 | yes |

Gaps versus the largest group as rate differences (+0.026 means 2.6 percentage points higher), with 95% bootstrap intervals:
| group | versus | flag_rate_gap | tpr_gap | fpr_gap |
|---|---|---|---|---|
| male | female | +0.026 [+0.004, +0.049] | +0.036 [-0.014, +0.097] | +0.013 [-0.006, +0.033] |

### EDUCATION
| group | n | default_rate | mean_pd | calib_ratio | auc | flag_rate | tpr | fpr | precision | reliable |
|---|---|---|---|---|---|---|---|---|---|---|
| university | 2,817 | 0.236 | 0.230 | 0.974 | 0.758 | 0.234 | 0.527 | 0.144 | 0.531 | yes |
| graduate school | 2,106 | 0.189 | 0.191 | 1.012 | 0.773 | 0.177 | 0.486 | 0.105 | 0.519 | yes |
| high school | 1,000 | 0.258 | 0.248 | 0.960 | 0.761 | 0.275 | 0.550 | 0.179 | 0.516 | yes |
| other | 77 | 0.104 | 0.141 | 1.355 | 0.551 | 0.091 | 0.125 | 0.087 | 0.143 | too small |

Gaps versus the largest group as rate differences (+0.026 means 2.6 percentage points higher), with 95% bootstrap intervals:
| group | versus | flag_rate_gap | tpr_gap | fpr_gap |
|---|---|---|---|---|
| graduate school | university | -0.057 [-0.080, -0.036] | -0.041 [-0.106, +0.015] | -0.039 [-0.061, -0.015] |
| high school | university | +0.041 [+0.007, +0.073] | +0.023 [-0.044, +0.089] | +0.036 [+0.008, +0.070] |

### MARRIAGE
| group | n | default_rate | mean_pd | calib_ratio | auc | flag_rate | tpr | fpr | precision | reliable |
|---|---|---|---|---|---|---|---|---|---|---|
| single | 3,174 | 0.210 | 0.221 | 1.050 | 0.768 | 0.225 | 0.531 | 0.144 | 0.497 | yes |
| married | 2,755 | 0.234 | 0.214 | 0.915 | 0.765 | 0.211 | 0.505 | 0.122 | 0.558 | yes |
| other | 71 | 0.211 | 0.237 | 1.124 | 0.693 | 0.225 | 0.400 | 0.179 | 0.375 | too small |

Gaps versus the largest group as rate differences (+0.026 means 2.6 percentage points higher), with 95% bootstrap intervals:
| group | versus | flag_rate_gap | tpr_gap | fpr_gap |
|---|---|---|---|---|
| married | single | -0.014 [-0.035, +0.005] | -0.027 [-0.081, +0.019] | -0.022 [-0.039, -0.005] |

### AGE
| group | n | default_rate | mean_pd | calib_ratio | auc | flag_rate | tpr | fpr | precision | reliable |
|---|---|---|---|---|---|---|---|---|---|---|
| 30-39 | 2,238 | 0.199 | 0.204 | 1.025 | 0.765 | 0.202 | 0.513 | 0.125 | 0.506 | yes |
| <30 | 1,932 | 0.232 | 0.237 | 1.020 | 0.767 | 0.249 | 0.550 | 0.158 | 0.512 | yes |
| 40-49 | 1,303 | 0.229 | 0.209 | 0.909 | 0.754 | 0.195 | 0.465 | 0.115 | 0.547 | yes |
| 50+ | 527 | 0.252 | 0.228 | 0.905 | 0.780 | 0.235 | 0.534 | 0.135 | 0.573 | yes |

Gaps versus the largest group as rate differences (+0.026 means 2.6 percentage points higher), with 95% bootstrap intervals:
| group | versus | flag_rate_gap | tpr_gap | fpr_gap |
|---|---|---|---|---|
| <30 | 30-39 | +0.047 [+0.023, +0.072] | +0.037 [-0.028, +0.105] | +0.033 [+0.010, +0.056] |
| 40-49 | 30-39 | -0.007 [-0.036, +0.020] | -0.049 [-0.123, +0.024] | -0.010 [-0.035, +0.014] |
| 50+ | 30-39 | +0.033 [-0.001, +0.074] | +0.020 [-0.076, +0.113] | +0.010 [-0.023, +0.047] |

## LightGBM (cutoff 0.36, flags 16.7% of customers)

### SEX
| group | n | default_rate | mean_pd | calib_ratio | auc | flag_rate | tpr | fpr | precision | reliable |
|---|---|---|---|---|---|---|---|---|---|---|
| female | 3,663 | 0.213 | 0.211 | 0.994 | 0.781 | 0.159 | 0.439 | 0.084 | 0.586 | yes |
| male | 2,337 | 0.234 | 0.226 | 0.964 | 0.777 | 0.178 | 0.458 | 0.092 | 0.603 | yes |

Gaps versus the largest group as rate differences (+0.026 means 2.6 percentage points higher), with 95% bootstrap intervals:
| group | versus | flag_rate_gap | tpr_gap | fpr_gap |
|---|---|---|---|---|
| male | female | +0.019 [-0.001, +0.037] | +0.019 [-0.038, +0.077] | +0.008 [-0.008, +0.025] |

### EDUCATION
| group | n | default_rate | mean_pd | calib_ratio | auc | flag_rate | tpr | fpr | precision | reliable |
|---|---|---|---|---|---|---|---|---|---|---|
| university | 2,817 | 0.236 | 0.228 | 0.969 | 0.773 | 0.188 | 0.477 | 0.099 | 0.598 | yes |
| graduate school | 2,106 | 0.189 | 0.189 | 1.002 | 0.790 | 0.122 | 0.375 | 0.063 | 0.582 | yes |
| high school | 1,000 | 0.258 | 0.250 | 0.970 | 0.773 | 0.211 | 0.488 | 0.115 | 0.597 | yes |
| other | 77 | 0.104 | 0.147 | 1.417 | 0.638 | 0.039 | 0.125 | 0.029 | 0.333 | too small |

Gaps versus the largest group as rate differences (+0.026 means 2.6 percentage points higher), with 95% bootstrap intervals:
| group | versus | flag_rate_gap | tpr_gap | fpr_gap |
|---|---|---|---|---|
| graduate school | university | -0.067 [-0.088, -0.047] | -0.102 [-0.167, -0.044] | -0.036 [-0.055, -0.017] |
| high school | university | +0.023 [-0.006, +0.052] | +0.011 [-0.055, +0.078] | +0.016 [-0.012, +0.043] |

### MARRIAGE
| group | n | default_rate | mean_pd | calib_ratio | auc | flag_rate | tpr | fpr | precision | reliable |
|---|---|---|---|---|---|---|---|---|---|---|
| single | 3,174 | 0.210 | 0.221 | 1.048 | 0.788 | 0.173 | 0.463 | 0.095 | 0.564 | yes |
| married | 2,755 | 0.234 | 0.213 | 0.910 | 0.774 | 0.160 | 0.433 | 0.077 | 0.633 | yes |
| other | 71 | 0.211 | 0.236 | 1.117 | 0.683 | 0.155 | 0.333 | 0.107 | 0.455 | too small |

Gaps versus the largest group as rate differences (+0.026 means 2.6 percentage points higher), with 95% bootstrap intervals:
| group | versus | flag_rate_gap | tpr_gap | fpr_gap |
|---|---|---|---|---|
| married | single | -0.013 [-0.032, +0.004] | -0.029 [-0.085, +0.015] | -0.019 [-0.035, -0.003] |

### AGE
| group | n | default_rate | mean_pd | calib_ratio | auc | flag_rate | tpr | fpr | precision | reliable |
|---|---|---|---|---|---|---|---|---|---|---|
| 30-39 | 2,238 | 0.199 | 0.201 | 1.010 | 0.786 | 0.147 | 0.437 | 0.075 | 0.591 | yes |
| <30 | 1,932 | 0.232 | 0.234 | 1.009 | 0.784 | 0.195 | 0.492 | 0.105 | 0.586 | yes |
| 40-49 | 1,303 | 0.229 | 0.212 | 0.926 | 0.762 | 0.155 | 0.395 | 0.084 | 0.584 | yes |
| 50+ | 527 | 0.252 | 0.232 | 0.921 | 0.776 | 0.173 | 0.444 | 0.081 | 0.648 | yes |

Gaps versus the largest group as rate differences (+0.026 means 2.6 percentage points higher), with 95% bootstrap intervals:
| group | versus | flag_rate_gap | tpr_gap | fpr_gap |
|---|---|---|---|---|
| <30 | 30-39 | +0.048 [+0.027, +0.070] | +0.055 [-0.004, +0.118] | +0.030 [+0.009, +0.048] |
| 40-49 | 30-39 | +0.008 [-0.018, +0.033] | -0.043 [-0.117, +0.027] | +0.008 [-0.011, +0.028] |
| 50+ | 30-39 | +0.025 [-0.011, +0.061] | +0.006 [-0.093, +0.103] | +0.006 [-0.022, +0.037] |
