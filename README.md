# Credit Origination Scorecard

## Overview

This project develops an interpretable **credit-origination scorecard** that estimates an applicant’s probability of significant payment difficulties during the 12 months following application.

The model converts predicted probabilities into a conventional credit score and supports a three-tier lending policy:

- automatic acceptance for low-risk applications;
- manual underwriting for borderline applications;
- automatic decline for higher-risk applications.

The project is a methodological demonstration using entirely **synthetic data**. It is not a production-ready lending model.

## Objectives

- Construct and clean a realistic synthetic application dataset.
- Define known-good, intermediate and known-bad outcomes.
- Compare alternative logistic-regression specifications.
- Develop an interpretable log-odds scorecard.
- Test performance on chronological holdout and through-the-door samples.
- Translate model predictions into a feasible lending policy.
- Evaluate the trade-off between portfolio risk, approval rates and underwriting workload.

## Methodology

1. Generated 10,000 synthetic applications containing application, affordability, credit-bureau and 12-month performance information.
2. Linked the application, bureau and performance records using account identifiers.
3. Removed unreliable links, invalid observations and variables unavailable at the point of application.
4. Classified applicants using missed-payment performance over a 12 month period:
   - **Known good:** no missed-payment months.
   - **Intermediate:** one or two missed-payment months.
   - **Known bad:** three or more missed-payment months.
5. Created a known-good/bad development sample and a chronologically later holdout sample.
6. Compared four logistic-regression specifications:
   - Delphi score only;
   - application and bureau variables;
   - variables plus Delphi;
   - a reduced seven-variable model.
7. Assessed discrimination, calibration and parsimony using AIC, log loss, Brier score, ROC AUC and Gini.
8. Converted predicted probabilities into a scorecard anchored at 600 points for good-to-bad odds of 50:1, with 20 points representing a doubling of the odds.
9. Reintroduced the intermediate population for policy and sensitivity analysis.
10. Tested cutoffs on a separate 2,000-account synthetic through-the-door cohort.
11. Stress-tested manual-underwriting accuracy, referral costs, accepted-bad costs and debt-to-income overrides in order to establish feasible policy recommendations for the tool.

## Final Model

The **reduced KGB logistic regression** was selected as the primary scorecard model. It contains seven terms:

- debt-to-income ratio;
- recent application count;
- revolving utilisation;
- historical defaults;
- log address tenure;
- low-disposable-income indicator;
- education loan-purpose indicator.

The Delphi score was excluded because it provided little additional predictive value once the selected application and bureau variables were included.

## Key Results

Four candidate models were evaluated on the same 2,458-account chronological holdout sample.

| Model | Terms | AIC | Brier score | ROC AUC | Gini |
|---|---:|---:|---:|---:|---:|
| Delphi only | 1 | 2,452.2 | 0.0397 | 0.5702 | 0.1405 |
| Variables only | 50 | 2,460.8 | 0.0396 | 0.6085 | 0.2171 |
| Variables and Delphi | 51 | 2,462.7 | 0.0396 | 0.6093 | 0.2186 |
| **Reduced KGB model** | **7** | **2,409.7** | **0.0394** | **0.6442** | **0.2884** |

The reduced model produced the strongest holdout discrimination and lowest AIC and Brier score while using substantially fewer predictors.

Its mean predicted holdout bad rate was **3.91%**, compared with an observed rate of **4.15%**.

### Through-the-door test

A separate 2025 cohort of 2,000 synthetic applicants was scored before its generated outcomes were joined back for evaluation.

At the proposed 2.50% acceptance cutoff:

| Result | Value |
|---|---:|
| Approval rate | 18.2% |
| Predicted accepted bad rate | 2.03% |
| Observed accepted bad rate | 2.75% |
| Bad capture in the flagged population | 87.5% |

The calibration gap indicated that the model was not sufficiently reliable for a completely automated accept-or-decline policy.

## Recommended Lending Policy

| Decision | Predicted probability of bad | Approximate score | Action |
|---|---:|---:|---|
| Auto accept | Below 2.50% | Above 592.8 | Accept, subject to eligibility and fraud checks |
| Manual review | 2.50% to below 3.50% | 582.8–592.8 | Refer to an underwriter |
| Auto decline | 3.50% or above | Below 582.8 | Decline, subject to policy requirements |

### Final results notebook

`05_final_results.ipynb` consolidates the final model and policy results across the development, chronological Out and through-the-door (TTD) samples. It reloads the three samples, refits the selected seven-term logistic-regression model on the development data, applies the confirmed 2.50% and 3.50% probability thresholds, and estimates the policy outcomes assuming 80% underwriting accuracy. Under that assumption, an underwriter correctly accepts 80% of reviewed good applicants and correctly declines 80% of reviewed bad applicants.

| Sample | Accounts | Assumed underwriting accuracy | Observed bad rate | Calibration gap | Auto accept rate | Manual review rate | Auto decline rate | Expected approval rate | Expected accepted bad rate | Expected good decline rate | Expected bad capture rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Development | 7,405 | 80% | 3.97% | +0.00% | 17.2% | 33.7% | 49.1% | 43.6% | 1.24% | 52.9% | 86.3% |
| Out | 2,508 | 80% | 4.07% | +0.24% | 18.3% | 34.4% | 47.2% | 45.2% | 1.09% | 51.2% | 87.8% |
| TTD | 2,000 | 80% | 4.00% | +0.15% | 18.2% | 35.9% | 45.9% | 46.1% | 1.65% | 50.6% | 81.0% |

The development sample is calibrated by construction. The positive calibration gaps of 0.24 percentage points on Out and 0.15 percentage points on TTD show modest underprediction of bad outcomes outside development. Policy allocation is reasonably stable across the samples: approximately 17–18% of applications are automatically accepted, 34–36% require manual review and 46–49% are automatically declined.

With 80% underwriting accuracy, the expected approval rate is 44–46% and the expected accepted bad rate remains below the 2% target in all three samples. TTD is the most cautious indicator of forward performance: its expected accepted bad rate is 1.65% and its expected bad capture rate is 81.0%. The policy is conservative, however, because approximately half of all applicants are expected to be good applicants who are declined, while more than one-third require manual review. These results support the thresholds as an initial monitored policy, but the calibration, underwriting assumption and good-customer cost require validation with real application and outcome data.

In the TTD sample using the policy assigned and assuming an 80% underwriter accuracy:

|Review accuracy|DTI override|Applicants reviewed|Review rate|Incremental reviews from DTI|Expected accepted accounts|Expected approval rate|Expected accepted bads|Expected accepted bad rate|Expected good declines|Expected bads prevented|Accepted-bad cost|Review cost|Good-decline cost|Total scenario cost|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|80%|>15%|782|39.1%|64|907.6|45.4%|12.0|1.32%|1,024.4|68.0|1,200.0|1,564.0|10,244.0|13,008.0|

### DTI override

Referring otherwise auto-accepted applicants with debt-to-income ratios above 15% identified four additional bad accounts among 64 referrals (40% share of all the automatically accepted bads). However, this result is based on a small synthetic sample and was not monotonic across alternative thresholds.

The rule should therefore be monitored in **shadow mode**, rather than used as a mandatory launch rule.

## Tech Stack

- Python
- JupyterLab
- Pandas
- NumPy
- Statsmodels
- Scikit-learn
- Matplotlib
- Pytest
- Ruff
- uv

## Project Structure

```text
Origination-Scorecard/
├── data/
│   ├── Original data/
│   ├── Data Cleaning/
│   ├── KGB Modelling/
│   ├── KIGB Modelling/
│   └── TTD Modelling/
├── notebooks/
│   ├── 01_sample_generation.ipynb
│   ├── 02_KGB_model.ipynb
│   ├── 03_KIGB_model.ipynb
│   ├── 04_TTD_testing.ipynb
│   └── 05_final_results.ipynb
├── scripts/
├── src/origination_scorecard/
├── tests/
├── MODEL_RESULTS_COMPARISON.md
├── pyproject.toml
└── uv.lock
```

## How to Run

Install the dependencies from the project directory:

```powershell
uv sync
```

Launch JupyterLab:

```powershell
uv run jupyter lab
```

Run the notebooks in numerical order:

1. `01_sample_generation.ipynb`
2. `02_KGB_model.ipynb`
3. `03_KIGB_model.ipynb`
4. `04_TTD_testing.ipynb`
5. `05_final_results.ipynb`

To run the automated tests:

```powershell
uv run pytest
```

## Assumptions and Limitations

- All applicants, outcomes and credit information are synthetic.
- Development and through-the-door samples were generated from related underlying processes.
- The analysis does not demonstrate performance under real economic conditions, fraud patterns or customer-selection effects.
- Underwriting accuracy and policy costs are illustrative scenario assumptions.
- The cutoffs are provisional operating thresholds rather than proven commercial optima.
- Live use would require genuine outcome data, independent validation, fairness testing, stability monitoring and documented model governance.
- Real exposure, recovery, loss and operational-cost data would be required to optimise the policy economically.

## Final Recommendation

Take the KGB model log odds scoring system and use the three tiered decisioning system to select outcomes for applicants; Auto accept, Manual review, Auto decline.

Monitor costs, ensure that underwriter decisions are recorded against using reason codes. This will help to define Underwriter accuracy over time, and to identify other variables that are key to decisioning.

The model currently predicts, given the current assumptions, that it will produce a conservative bad rate on the book of less than the target: 2%. While the model is in its infancy, It is best to be conservative and monitor model drift over time. The conservative prediction gives the headroom, and means that the model and its policies can be adjusted over time once the realised results materialise to improve its ability to discuminate between goods and bads, without risking too many bads on the books in its infancy.

Do not relax the thresholds or implement the DTI override as a mandatory rule until the model has been calibrated and independently validated using genuine application and performance data.

## Disclaimer

This project is for educational and research purposes only. It should not be used to make real lending decisions without appropriate data, validation, governance and regulatory review.
