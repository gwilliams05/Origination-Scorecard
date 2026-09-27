"""Fit the reduced KGB model with Delphi and the three derived variables."""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

# Works from either the project folder or its notebooks folder.
relative = Path("data") / "KGB Modelling"
root = next(
    (p for p in [Path.cwd(), *Path.cwd().parents]
     if (p / relative / "KGB in sample.csv").is_file()),
    None,
)
if root is None:
    raise FileNotFoundError("Cannot locate the KGB sample files.")

numeric = [
    "committed_monthly_outgoings", "debt_to_income_ratio",
    "recent_application_count", "revolving_utilisation", "historic_defaults",
    "DelphiScore",
]
required = [
    "target_bad", "loan_purpose", *numeric, "disposable_income",
    "time_at_address_months", "employment_tenure_months",
]

def load_sample(filename):
    data = pd.read_csv(root / relative / filename)
    rows = data[required].copy().replace(r"^\s*$", np.nan, regex=True)
    for column in set(required) - {"loan_purpose"}:
        rows[column] = pd.to_numeric(rows[column], errors="raise")
    rows = rows.replace([np.inf, -np.inf], np.nan).dropna().copy()
    print(f"{filename}: retained {len(rows):,}; excluded {len(data) - len(rows):,}")
    if not rows["target_bad"].isin([0, 1]).all() or rows["target_bad"].nunique() != 2:
        raise ValueError(f"{filename}: target must contain both 0 and 1.")
    if (rows[["time_at_address_months", "employment_tenure_months"]] < 0).any().any():
        raise ValueError(f"{filename}: tenure cannot be negative.")
    rows["low_disposable_income_flag"] = (rows["disposable_income"] < 150).astype(float)
    rows["log_address_tenure"] = np.log1p(rows["time_at_address_months"])
    rows["employment_address_instability_count"] = (
        (rows["employment_tenure_months"] < 12).astype(float)
        + (rows["time_at_address_months"] < 12).astype(float)
    )
    return rows

train = load_sample("KGB in sample.csv")
test = load_sample("KGB out sample.csv")

# Estimate all scaling parameters on training data only.
scaled = [*numeric, "log_address_tenure"]
means = train[scaled].mean()
stds = train[scaled].std(ddof=0)
if (stds == 0).any():
    raise ValueError("A training predictor has no variation.")

def design_matrix(rows):
    design = (rows[scaled] - means) / stds
    design.insert(0, "const", 1.0)
    for column in ["low_disposable_income_flag", "employment_address_instability_count"]:
        design[column] = rows[column]
    design["loan_purpose=education"] = (rows["loan_purpose"] == "education").astype(float)
    return design.astype(float)

X_train, X_test = design_matrix(train), design_matrix(test)
y_train, y_test = train["target_bad"].astype(int), test["target_bad"].astype(int)
if np.linalg.matrix_rank(X_train.to_numpy()) < X_train.shape[1]:
    raise ValueError("Training predictors are perfectly collinear.")

model = sm.Logit(y_train, X_train).fit(maxiter=200, disp=False)
if not model.mle_retvals.get("converged", False):
    raise RuntimeError("Model did not converge.")
if not np.isfinite(model.params).all() or not np.isfinite(model.bse).all():
    raise RuntimeError("Invalid coefficients or standard errors.")

train_pd, test_pd = model.predict(X_train), model.predict(X_test)

def performance(actual, probability):
    auc = roc_auc_score(actual, probability)
    fpr, tpr, _ = roc_curve(actual, probability)
    return {
        "Accounts": len(actual),
        "Bad rate": actual.mean(),
        "Mean predicted PD": probability.mean(),
        "Brier": brier_score_loss(actual, probability),
        "ROC AUC": auc,
        "Gini": 2 * auc - 1,
        "KS": np.max(tpr - fpr),
        "Log loss": log_loss(actual, probability),
    }

results = pd.DataFrame({
    "In-sample": performance(y_train, train_pd),
    "Out-of-sample": performance(y_test, test_pd),
})
print("\nREDUCED MODEL + DELPHI + DERIVED VARIABLES")
print(results.to_string(float_format=lambda value: f"{value:.4f}"))
print(f"\nTraining AIC: {model.aic:.2f}")
print(f"Training McFadden pseudo R-squared: {model.prsquared:.4f}")
print(model.summary())
