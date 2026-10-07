# Model results comparison

## Executive conclusion

The **Reduced final KGB model** is the preferred model for constructing the scorecard and log-odds points system. It produced the strongest chronological holdout discrimination of the four directly comparable KGB candidates, while using only seven predictor terms. The KIGB refit is useful as a sensitivity and policy model, but its results are not directly comparable with the KGB candidates because it changes both the sample and the target definition by treating indeterminate cases as non-bads.

## 1. KGB candidate-model comparison

All four models below were developed using the same 7,258 known-good/known-bad development accounts and assessed on the same 2,458-account chronological holdout. The holdout contains 102 observed bads, giving an observed bad rate of 4.15%.

| Model | Terms | Development AIC | Holdout observed bad rate | Holdout mean predicted PD | Calibration gap | Log loss | Brier score | ROC AUC | Gini |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Delphi only | 1 | 2,452.2 | 4.15% | 4.03% | +0.12% | 0.1716 | 0.0397 | 0.5702 | 0.1405 |
| Variables only | 50 | 2,460.8 | 4.15% | 3.92% | +0.23% | 0.1705 | 0.0396 | 0.6085 | 0.2171 |
| Variables + Delphi | 51 | 2,462.7 | 4.15% | 3.92% | +0.23% | 0.1704 | 0.0396 | 0.6093 | 0.2186 |
| **Reduced final KGB model** | **7** | **2,409.7** | **4.15%** | **3.91%** | **+0.24%** | **0.1677** | **0.0394** | **0.6442** | **0.2884** |

### Interpretation

- **Reduced final KGB model:** best overall result. It has the lowest AIC, log loss and Brier score, and the highest holdout ROC AUC and Gini. It also achieves this with substantially fewer terms than the two full application models.
- **Variables + Delphi:** Delphi adds almost no incremental discrimination over the variables-only model. Holdout ROC AUC rises by only 0.0008, while model complexity increases.
- **Variables only:** performs better than Delphi alone, but materially worse than the reduced model despite containing 50 terms.
- **Delphi only:** is the simplest benchmark, but has the weakest discriminatory power.

## 2. Final reduced KGB model versus the KIGB refit

| Model and sample | Accounts | Target treatment | Observed bad rate | Mean predicted PD | Calibration gap | Brier score | ROC AUC | Gini |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Reduced KGB — chronological holdout | 2,458 | Known goods and known bads only | 4.15% | 3.91% | +0.24% | 0.0394 | 0.6442 | 0.2884 |
| KIGB refit — development | 7,405 | Indeterminates included as `kigb_bad = 0` | 3.97% | 3.97% | +0.00% | 0.0378 | 0.6385 | 0.2769 |
| KIGB refit — chronological holdout | 2,508 | Indeterminates included as `kigb_bad = 0` | 4.07% | 3.83% | +0.24% | 0.0386 | 0.6429 | 0.2858 |

The KIGB holdout statistics look similar to those of the reduced KGB model, but this is not a like-for-like challenger test. The KIGB model adds 50 holdout indeterminates and labels them as non-bads. Its lower observed bad rate and Brier score are therefore partly affected by the revised target population. The KGB model remains the cleaner basis for estimating risk and forming the log-odds score.

## 3. Through-the-door test of the KIGB implementation

| Sample | Accounts | Expected bads | Observed bads | Predicted bad rate | Observed bad rate | Calibration gap | Brier score | ROC AUC | Gini |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Simulated 2025 TTD cohort | 2,000 | 77.1 | 80 | 3.85% | 4.00% | +0.15% | 0.0383 | 0.5887 | 0.1774 |

The TTD test remains reasonably calibrated at portfolio level, underpredicting the observed bad rate by 0.15 percentage points. Discrimination is weaker than in the development and chronological holdout samples: ROC AUC falls to 0.5887 and Gini to 0.1774. This should be treated as evidence of performance deterioration under a changed sample rather than as a direct candidate-model comparison.

## Metric guide

| Metric | Preferred direction | Meaning |
|---|---|---|
| AIC | Lower | Development fit after penalising additional parameters; only comparable for models fitted to the same target and sample. |
| Log loss | Lower | Penalises inaccurate probability estimates, particularly confident errors. |
| Brier score | Lower | Mean squared error of predicted probabilities. |
| ROC AUC | Higher | Ability to rank bad accounts above good accounts. |
| Gini | Higher | Rescaled AUC, calculated as `2 × AUC − 1`. |
| Calibration gap | Closer to zero | Observed bad rate minus mean predicted PD; a positive value indicates underprediction. |

## Recommendation

Retain the **Reduced final KGB model** as the primary scorecard model and use its fitted regression to construct the log-odds points scale. Treat the KIGB refit and TTD results as sensitivity, policy and monitoring evidence rather than as replacements for the clean known-good/known-bad regression.

## Source

Results were taken from the executed outputs in:

- `notebooks/02_KGB_model.ipynb`
- `notebooks/03_KIGB_model.ipynb`
- `notebooks/04_TTD_testing.ipynb`

