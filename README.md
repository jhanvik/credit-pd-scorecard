# Credit Card Default (PD) Scorecard

## 1. Business question
For each existing credit-card customer, estimate the probability that they will
default on next month's payment (PD), then choose a cutoff for a risk action
(e.g. outreach or a credit-limit review) and show its expected business impact.
This is a behavioral model for existing customers, not an application scorecard.

## 2. Data
- Source: UCI Machine Learning Repository, "Default of Credit Card Clients" (dataset 350)
  https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- License: CC BY 4.0 (per the UCI page).
- Citation: Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C55S3H
- 30,000 customers, 23 input columns plus ID and target. No missing values.
  Amounts are in NT dollars.
- The data file is not stored in this repo. See "How to run".

## 3. Target
`default payment next month` (renamed DEFAULT): 1 = default, 0 = no default.
Default rate: 22.12% (6,636 of 30,000). This is much higher than a typical real loan book.
The dataset description does not name the month being predicted.

## 4. Observation window
Monthly records from April to September 2005. In the data, the "1" columns
(PAY_1, BILL_AMT1, PAY_AMT1) are September 2005, "2" is August, and so on
down to "6" = April 2005. PAY_1 is named PAY_0 in the original file and is
renamed here for consistency.

## 5. Split
Stratified 60/20/20 train/validation/test, fixed random seed. The test set is
locked and touched once at the end. The data has no calendar dates for each
customer, so an out-of-time split is not possible (see Limitations).

## 6. Metrics
AUC, Gini, KS, precision-recall, Brier score and calibration curve.
A scorecard (WoE + logistic regression) is compared honestly with a LightGBM challenger.

## 7. Known data issues (decisions documented in Phase 2)
- EDUCATION is documented as 1-4, but the data also contains 0 (14 rows),
  5 (280) and 6 (51). MARRIAGE is documented as 1-3, but also contains 0 (54).
- PAY_x is documented as -1 (paid duly) and 1-9 (months late). The data also
  contains -2 and 0, which are not documented. In September, 0 covers about 49%
  of customers and -2 about 9%. They are treated as separate categories until
  their meaning is verified.
- 56 rows repeat the features of an earlier row (52 groups, ignoring ID and target); in 21 of those groups
  the outcomes disagree (label noise). 1,930 rows have a negative BILL_AMT.
- SEX, EDUCATION, MARRIAGE and AGE are demographic: all four are excluded from the model inputs
  and used only in the later fairness check (verify fair-lending rules on age for your use).

## 8. Limitations
- Single table, one cohort of Taiwanese credit-card customers; may not generalize.
- No time dimension for out-of-time validation or look-ahead testing.
- Undocumented codes in PAY_x, EDUCATION and MARRIAGE limit interpretability.
- Business-impact numbers use illustrative, labeled assumptions, not real costs.

## 9. How to run
Setup and run instructions will be added in the final phase.
