# Credit Card Default (PD) Scorecard

[![CI](https://github.com/jhanvik/credit-pd-scorecard/actions/workflows/ci.yml/badge.svg)](https://github.com/jhanvik/credit-pd-scorecard/actions/workflows/ci.yml)

An end-to-end probability-of-default (PD) model: audit, leakage check, WoE/IV binning, a
points-based logistic scorecard, a LightGBM challenger, explainability, stability, a fairness
check, a cost-based cutoff, a Streamlit demo, and a one-time locked test-set evaluation.

## Results at a glance

Final, one-time evaluation on the locked test split (6,000 customers, 1,327 defaults).
Models, bins, features and cutoffs were frozen on training/validation data before this run.
Nothing was refitted or retuned afterwards. Intervals are 95% bootstrap (1,000 resamples).

| Model | Test AUC (95% CI) | Gini | KS | Brier | Validation AUC |
|---|---|---|---|---|---|
| WoE logistic scorecard (12 features) | 0.7717 (0.7573 - 0.7867) | 0.5433 | 0.4240 | 0.1357 | 0.7648 |
| LightGBM challenger | 0.7919 (0.7784 - 0.8058) | 0.5837 | 0.4479 | 0.1314 | 0.7798 |

- LightGBM minus scorecard, test AUC: **+0.0202** (paired bootstrap 95% interval +0.0137 to +0.0268).
  On validation the gap was +0.0150 (+0.0085 to +0.0217). The ranking advantage is consistent.
- Test results were slightly higher than validation for both models. That is within sampling
  noise and should not be read as the models improving.
- Calibration by risk decile on test is close to the diagonal for both models
  (see `reports/final_test_report.md`).

### Business impact is a tie (illustrative assumptions)

At the cutoffs frozen on validation (scorecard 0.30, LightGBM 0.36), under the labeled assumptions
below, the two models are not distinguishable on net benefit:

| | Test net benefit (NT$, 95% CI) | Customers treated | Precision | Recall |
|---|---|---|---|---|
| Scorecard @ 0.30 | 3,735,309 (3,123,625 - 4,348,015) | 22.7% | 0.535 | 0.550 |
| LightGBM @ 0.36 | 3,714,898 (3,124,424 - 4,323,257) | 17.4% | 0.614 | 0.482 |

Paired difference (LightGBM minus scorecard): **NT$ -20,411, 95% interval -212,595 to +164,658**.
The interval includes zero, and on validation the sign was reversed (+37,033). So the better ranker
did not demonstrably earn more. Both beat "treat nobody" (0) and "treat everybody" (324,058).

Assumptions (illustrative, NOT real costs): LGD 0.60; the action avoids 20% of the loss for treated
defaulters; outreach cost NT$100 per treated customer; false-alarm cost NT$1,500; exposure =
max(latest bill, 0). The best cutoff moves a lot across plausible assumptions (0.06 to 0.48 in the
sensitivity sweep), so treat the absolute net-benefit figures as a demonstration of the method.
The absolute figures are also higher on test than on validation (about NT$3.7M vs NT$2.7M);
I have not established why and do not rely on them.

## Validation report (self-validation)
I also wrote a model validation report in the style of a bank's model-risk review, structured around
the themes of SR 26-2 (conceptual soundness, outcomes analysis, ongoing monitoring). It is a
**self-validation**: I built the model and wrote the report, so independence is limited, and the report
lists that as its first finding.

- Outcome: approve with conditions, for demonstration use only. 14 findings (2 High, 7 Medium, 5 Low).
- Report: [`validation/validation_report.pdf`](validation/validation_report.pdf) (16 pages).
- Scripts and results: [`validation/`](validation/) (`replicate.py`, `outcomes.py`, `stress.py` and their
  `*_results.md` outputs). These read validation data only; the consumed test split is not touched.
- Rating and threshold choices in the report are my own assumptions, labeled as such.

## 1. Business question
For each existing credit-card customer, estimate the probability that they will default on next
month's payment (PD), then choose a cutoff for a risk action (e.g. outreach or a credit-limit
review) and show its expected business impact. This is a behavioral model for existing customers,
not an application scorecard.

## 2. Data
- Source: UCI Machine Learning Repository, "Default of Credit Card Clients" (dataset 350)
  https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- License: CC BY 4.0 (per the UCI page).
- Citation: Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C55S3H
- 30,000 customers, 23 input columns plus ID and target. No missing values. Amounts are in NT dollars.
- The data file is not stored in this repo (`data/` is git-ignored). See "How to run".

## 3. Target and window
`default payment next month` (renamed DEFAULT): 1 = default. Default rate 22.12% (6,636 of 30,000),
much higher than a typical real loan book. Monthly records run April to September 2005; the "1"
columns (PAY_1, BILL_AMT1, PAY_AMT1) are September. PAY_1 is named PAY_0 in the original file.
The dataset description does not name the outcome month.

## 4. Method
1. **Audit** (`reports/data_audit.md`): shapes, duplicates, undocumented codes, class balance.
2. **Split**: stratified 60/20/20 (seed 42): 18,000 / 6,000 / 6,000 rows, each about 22.12% default.
   Row-ID hashes are recorded in a manifest, and test access requires an explicit confirmation flag.
3. **Leakage check** (`reports/leakage_table.md`): a written decision for every column plus a
   single-feature AUC screen (flag above 0.75; none flagged, highest 0.731). Demographics are
   excluded from the model.
4. **Features**: 19 raw behavioral inputs plus 10 engineered (utilization, payment ratios, delinquency counts).
5. **WoE/IV binning** fitted on train only, with monotonic bins where sensible. IV bands are
   conventions, not laws.
6. **Scorecard**: IV filter, near-duplicate and VIF pruning (VIF below 5), logistic regression with
   iterative removal of wrong-sign or insignificant terms. Result: 12 features, all coefficients
   negative, max VIF 2.90. Points scaling: base 600 at 50:1 good:bad odds, 20 points to double
   the odds (a common convention, chosen for illustration).
7. **Challenger**: LightGBM, 20-draw randomized search with 5-fold stratified CV inside train only.
8. **Explainability**: SHAP global/local for LightGBM; reason codes for the scorecard.
9. **Stability**: PSI of score and features.
10. **Fairness check**: outcomes by SEX, EDUCATION, MARRIAGE, AGE (not model inputs).
11. **Cutoffs** chosen on validation by expected net benefit, then frozen.
12. **Test**: evaluated once, using the frozen artifacts (`reports/final_test_report.md`).
    `reports/TEST_SET_USED.json` records the run; the command refuses to run again while it exists.

## 5. Fairness and stability notes
- Score PSI validation to test is 0.0018. The split is random, so near-zero PSI is expected by
  construction and does NOT show stability over time.
- On validation, the scorecard flagged under-30s noticeably more often than the largest age group
  (roughly 4.7 percentage points) with a false-positive rate about 3 points higher, and there are
  gaps by education. Calibration within groups was close to 1. Base rates differ by group, no legal
  threshold is claimed, and the rules that apply to a real lender are a compliance question
  (verify with compliance). See `reports/fairness_report.md`.

## 6. Known data issues
- EDUCATION is documented as 1-4 but also contains 0 (14 rows), 5 (280) and 6 (51). MARRIAGE is
  documented as 1-3 but also contains 0 (54).
- PAY_x is documented as -1 and 1-9, but the data also contains -2 and 0 (undocumented; in
  September, 0 is about 49% of customers and -2 about 9%). They are treated as separate categories.
- 56 rows repeat the features of an earlier row (52 groups); in 21 groups the outcomes disagree
  (label noise). 1,930 rows have a negative BILL_AMT.

## 7. Limitations
- One cohort of Taiwanese credit-card customers from 2005; may not generalize.
- Random split only. The data has no per-customer dates, so there is no out-of-time test, no
  look-ahead testing, and no real evidence of stability over time.
- Residual timing risk: if the September repayment status were recorded after the outcome window
  opened, PAY_1 would leak. The dataset text does not say. The delinquency features have very high
  IV, which is consistent with strong signal but also worth this caution.
- Undocumented codes in PAY_x, EDUCATION and MARRIAGE limit interpretability.
- Business-impact numbers use illustrative assumptions; the optimal cutoff is sensitive to them.
- Cutoffs were chosen on the validation split, so validation net benefit is optimistic by
  construction. The test figures are the unbiased ones.
- The challenger is statistically better on AUC but not demonstrably better on business value.
- The Dockerfile has not been built or run by the author (untested).
- Fairness: only a sanity check on four attributes, no causal analysis and no legal conclusion.

## 8. How to run
Requires macOS or Linux, [uv](https://docs.astral.sh/uv/) and git.

```zsh
git clone https://github.com/jhanvik/credit-pd-scorecard.git
cd credit-pd-scorecard
uv sync --locked                      # creates .venv with pinned dependencies
```

Download the UCI file (see Data) and place it at `data/raw/default of credit card clients.xls`, then:

```zsh
uv run python -m credit_pd_scorecard.load_data
uv run python -m credit_pd_scorecard.audit
uv run python -m credit_pd_scorecard.split
uv run python -m credit_pd_scorecard.build_features
uv run python -m credit_pd_scorecard.eda
uv run python -m credit_pd_scorecard.leakage
uv run python -m credit_pd_scorecard.fit_scorecard
uv run python -m credit_pd_scorecard.challenger
uv run python -m credit_pd_scorecard.explain
uv run python -m credit_pd_scorecard.stability
uv run python -m credit_pd_scorecard.business
uv run python -m credit_pd_scorecard.fairness
uv run python -m credit_pd_scorecard.export_artifacts
```

Check the code, then rehearse the final evaluation on validation data:

```zsh
uv run pytest -q
uv run ruff check .
uv run python -m credit_pd_scorecard.final_test --rehearsal
```

The one-time test evaluation (it refuses to run twice; do not use it for tuning):

```zsh
uv run python -m credit_pd_scorecard.final_test --confirm-final-run
```

Demo app (scores one customer; shows scorecard PD, score, LightGBM PD and reasons):

```zsh
uv run streamlit run app/streamlit_app.py
```

The scorecard and LightGBM runs are seeded. Regenerating the models should reproduce the reported
numbers, but if the library versions differ from `uv.lock` small differences are possible.

## 9. Project structure
```
src/credit_pd_scorecard/   pipeline modules (config, split, features, woe, scorecard, challenger, ...)
tests/                     pytest suite (82 tests; synthetic data with hand-checked answers)
app/streamlit_app.py       demo
artifacts/                 frozen scorecard, WoE bins, LightGBM model, cutoffs
reports/                   generated reports and figures
.github/workflows/ci.yml   lint + tests on every push
```
