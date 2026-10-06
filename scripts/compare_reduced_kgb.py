"""Compare the original model with a refit excluding three predictors."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Reuse the original preprocessing, samples, scaling and fitting settings.
original = runpy.run_path(str(Path(__file__).with_name("run_reduced_kgb.py")))
removed = [
    "DelphiScore",
    "employment_address_instability_count",
    "committed_monthly_outgoings",
]
X_train = original["X_train"].drop(columns=removed)
X_test = original["X_test"].drop(columns=removed)
y_train, y_test = original["y_train"], original["y_test"]
performance = original["performance"]

if np.linalg.matrix_rank(X_train.to_numpy()) < X_train.shape[1]:
    raise ValueError("Training predictors are perfectly collinear.")
model = sm.Logit(y_train, X_train).fit(maxiter=200, disp=False)
if not model.mle_retvals.get("converged", False):
    raise RuntimeError("Model did not converge.")
if not np.isfinite(model.params).all() or not np.isfinite(model.bse).all():
    raise RuntimeError("Invalid coefficients or standard errors.")

train_pd, test_pd = model.predict(X_train), model.predict(X_test)
comparison = pd.DataFrame(
    {
        "Original: in-sample": performance(y_train, original["train_pd"]),
        "Refit: in-sample": performance(y_train, train_pd),
        "Original: out-of-sample": performance(y_test, original["test_pd"]),
        "Refit: out-of-sample": performance(y_test, test_pd),
    }
)
print("\nCOMPARISON: REFIT EXCLUDES DELPHI, INSTABILITY AND COMMITTED OUTGOINGS")
print(comparison.to_string(float_format=lambda value: f"{value:.6f}"))
fit_statistics = pd.DataFrame(
    {
        "Original": {
            "Training AIC": original["model"].aic,
            "Training McFadden pseudo R-squared": original["model"].prsquared,
        },
        "Refit": {
            "Training AIC": model.aic,
            "Training McFadden pseudo R-squared": model.prsquared,
        },
    }
)
print(fit_statistics.to_string(float_format=lambda value: f"{value:.6f}"))
print(model.summary())
