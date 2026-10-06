"""Build and execute the candidate-model comparison notebook."""

import contextlib
import io
from pathlib import Path

import nbformat

root = Path(__file__).resolve().parents[1]
cells = []


def section(title: str, code: str) -> None:
    cells.append(nbformat.v4.new_markdown_cell(title))
    cells.append(nbformat.v4.new_code_cell(code.strip()))


cells.append(
    nbformat.v4.new_markdown_cell("""# Candidate model comparison

Refit seven candidate specifications on the same in-sample accounts and evaluate them on the same out-of-sample accounts. All scaling, category definitions and Delphi bucket boundaries come from training data. The reduced specification is fixed to the six previously selected terms; selection is not repeated.

**Lower Brier score and training AIC are better. Higher AUC and Gini are better.** AIC describes the training fit and is not an out-of-sample metric. This notebook recomputes comparable results, which can differ from historic results if those used different accounts. Exploring several models on the holdout makes it a model-comparison sample rather than a fresh final validation set.""")
)

section(
    "## Cell 1 — Import libraries and open both datasets",
    """
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from IPython.display import display
from sklearn.metrics import brier_score_loss, roc_auc_score

relative_dir = Path("data") / "KGB Modelling"
project_root = next(
    (folder for folder in [Path.cwd(), *Path.cwd().parents]
     if (folder / relative_dir / "KGB in sample.csv").is_file()),
    None,
)
if project_root is None:
    raise FileNotFoundError("Could not locate the KGB modelling folder.")

data_dir = project_root / relative_dir
kgb_in_sample = pd.read_csv(data_dir / "KGB in sample.csv")
kgb_out_sample = pd.read_csv(data_dir / "KGB out sample.csv")
target = "target_bad"
print(f"Loaded {len(kgb_in_sample):,} in-sample and {len(kgb_out_sample):,} out-of-sample accounts.")
""",
)

section(
    "## Cell 2 — Define inputs and establish common comparison samples",
    """
application_numeric = [
    "age_at_application", "employment_tenure_months",
    "time_at_address_months", "household_size", "dependants",
    "annual_gross_income", "estimated_monthly_net_income",
    "committed_monthly_outgoings", "rent_or_mortgage",
    "disposable_income", "debt_to_income_ratio", "requested_loan_amount",
    "bank_account_tenure_months", "application_hour",
    "recent_application_count", "credit_history_months",
    "active_credit_accounts", "revolving_utilisation",
    "recent_credit_searches", "historic_defaults", "historic_arrears",
]
application_categorical = [
    "employment_status", "occupation_group", "residential_status",
    "loan_purpose", "income_verification_result", "application_channel",
    "prior_customer_flag", "device_verification_result",
    "contact_verification_result", "electoral_roll_match",
]
required = [target, "DelphiScore", *application_numeric, *application_categorical]

def prepare_sample(data, name):
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    rows = data[required].copy().replace(r"^\\s*$", np.nan, regex=True)
    for column in [target, "DelphiScore", *application_numeric]:
        rows[column] = pd.to_numeric(rows[column], errors="raise")
    rows = rows.replace([np.inf, -np.inf], np.nan).dropna()
    if rows.empty or not rows[target].isin([0, 1]).all():
        raise ValueError(f"{name}: target must contain complete 0/1 values.")
    if rows[target].nunique() != 2:
        raise ValueError(f"{name}: both good and bad accounts are required.")
    for column in ["time_at_address_months", "employment_tenure_months"]:
        if rows[column].lt(0).any():
            raise ValueError(f"{name}: negative values in {column}.")
    for column in application_categorical:
        rows[column] = rows[column].astype(str)
    print(f"{name}: retained {len(rows):,}; excluded {len(data) - len(rows):,}.")
    return rows

train = prepare_sample(kgb_in_sample, "In-sample")
test = prepare_sample(kgb_out_sample, "Out-of-sample")
y_train = train[target].astype(int)
y_test = test[target].astype(int)

for column in application_categorical:
    unseen = set(test[column]) - set(train[column])
    if unseen:
        raise ValueError(f"Unseen out-of-sample categories in {column}: {sorted(unseen)}")

display(pd.DataFrame([
    {"Sample": name, "Accounts": len(rows),
     "Bads": int(rows[target].sum()), "Bad rate": rows[target].mean()}
    for name, rows in [("In-sample", train), ("Out-of-sample", test)]
]).style.format({"Bad rate": "{:.2%}"}))
""",
)

section(
    "## Cell 3 — Create the three derived variables",
    """
for rows in [train, test]:
    rows["low_disposable_income_flag"] = (rows["disposable_income"] < 150).astype(float)
    rows["log_address_tenure"] = np.log1p(rows["time_at_address_months"])
    rows["employment_address_instability_count"] = (
        (rows["employment_tenure_months"] < 12).astype(float)
        + (rows["time_at_address_months"] < 12).astype(float)
    )

derived = [
    "low_disposable_income_flag", "log_address_tenure",
    "employment_address_instability_count",
]
display(train[derived].head())
""",
)

section(
    "## Cell 4 — Build matching training and validation matrices",
    """
def make_matrices(numeric, categorical=(), unscaled=()):
    a = pd.DataFrame({"const": 1.0}, index=train.index)
    b = pd.DataFrame({"const": 1.0}, index=test.index)
    for column in numeric:
        mean = train[column].mean()
        std = train[column].std(ddof=0)
        if std == 0:
            continue
        a[column] = (train[column] - mean) / std
        b[column] = (test[column] - mean) / std
    for column in unscaled:
        a[column] = train[column]
        b[column] = test[column]
    for column in categorical:
        levels = sorted(train[column].unique())
        for level in levels[1:]:
            term = f"{column}={level}"
            a[term] = (train[column] == level).astype(float)
            b[term] = (test[column] == level).astype(float)
    return a.astype(float), b.astype(float)

reduced_numeric = [
    "committed_monthly_outgoings", "debt_to_income_ratio",
    "recent_application_count", "revolving_utilisation", "historic_defaults",
]

def reduced_matrices(with_delphi=False, with_derived=False):
    numeric = reduced_numeric.copy()
    if with_delphi:
        numeric.append("DelphiScore")
    unscaled = []
    if with_derived:
        numeric.append("log_address_tenure")
        unscaled = ["low_disposable_income_flag", "employment_address_instability_count"]
    a, b = make_matrices(numeric, unscaled=unscaled)
    for design, rows in [(a, train), (b, test)]:
        design["loan_purpose=education"] = (rows["loan_purpose"] == "education").astype(float)
    return a, b

# Learn five equal-frequency Delphi buckets from training scores only.
_, bucket_edges = pd.qcut(train["DelphiScore"], q=5, retbins=True, duplicates="drop")
if len(bucket_edges) < 3:
    raise ValueError("At least two Delphi buckets are required.")
bucket_edges[0], bucket_edges[-1] = -np.inf, np.inf
train_buckets = pd.cut(train["DelphiScore"], bins=bucket_edges, labels=False)
test_buckets = pd.cut(test["DelphiScore"], bins=bucket_edges, labels=False)
a_bucket = pd.DataFrame({"const": 1.0}, index=train.index)
b_bucket = pd.DataFrame({"const": 1.0}, index=test.index)
for bucket in range(1, len(bucket_edges) - 1):
    a_bucket[f"Delphi_bucket_{bucket + 1}"] = (train_buckets == bucket).astype(float)
    b_bucket[f"Delphi_bucket_{bucket + 1}"] = (test_buckets == bucket).astype(float)

candidate_matrices = {
    "Delphi only — continuous": make_matrices(["DelphiScore"]),
    "Delphi only — bucketed": (a_bucket, b_bucket),
    "Full application model": make_matrices(application_numeric, application_categorical),
    "Full application model + Delphi": make_matrices(
        [*application_numeric, "DelphiScore"], application_categorical
    ),
    "Reduced application model": reduced_matrices(),
    "Reduced model + Delphi": reduced_matrices(with_delphi=True),
    "Reduced model + Delphi + derived variables": reduced_matrices(
        with_delphi=True, with_derived=True
    ),
}
""",
)

section(
    "## Cell 5 — Fit every candidate on training data and calculate metrics",
    """
def remove_dependent_terms(a, b):
    # Remove exact dependencies, such as duplicate search counts and the
    # disposable-income identity. Apply the same retained columns to test.
    kept = []
    removed = []
    for column in a.columns:
        proposed = [*kept, column]
        if np.linalg.matrix_rank(a[proposed].to_numpy()) == len(proposed):
            kept.append(column)
        else:
            removed.append(column)
    return a[kept], b[kept], removed

fitted_models = {}
model_predictions = {}
results = []

for name, (a, b) in candidate_matrices.items():
    a, b, removed = remove_dependent_terms(a, b)
    fitted = sm.Logit(y_train, a).fit(method="newton", maxiter=200, disp=False)
    if not fitted.mle_retvals.get("converged", False):
        raise RuntimeError(f"{name}: regression did not converge.")
    if not np.isfinite(fitted.params).all() or not np.isfinite(fitted.bse).all():
        raise RuntimeError(f"{name}: coefficients or standard errors are invalid.")

    train_pd = fitted.predict(a)
    test_pd = fitted.predict(b)
    train_auc = roc_auc_score(y_train, train_pd)
    test_auc = roc_auc_score(y_test, test_pd)

    fitted_models[name] = fitted
    model_predictions[name] = {"In-sample": train_pd, "Out-of-sample": test_pd}
    results.append({
        "Model": name,
        "Parameters": len(fitted.params),
        "AIC (in-sample)": fitted.aic,
        "Brier (in-sample)": brier_score_loss(y_train, train_pd),
        "AUC (in-sample)": train_auc,
        "Gini (in-sample)": 2 * train_auc - 1,
        "Brier (out-of-sample)": brier_score_loss(y_test, test_pd),
        "AUC (out-of-sample)": test_auc,
        "Gini (out-of-sample)": 2 * test_auc - 1,
    })
    print(f"Fitted: {name}")
    if removed:
        print(f"  Removed dependent terms: {', '.join(removed)}")

comparison_table = pd.DataFrame(results).set_index("Model")
""",
)

section(
    "## Cell 6 — Display the comparison table",
    """
print("Lower Brier and AIC are better. Higher AUC and Gini are better.")
print("All models use the same training accounts and the same validation accounts.")

formatting = {
    column: ("{:,.0f}" if column == "Parameters" else
             "{:,.2f}" if column == "AIC (in-sample)" else "{:.4f}")
    for column in comparison_table.columns
}
display(comparison_table.style.format(formatting))

# Optional export:
# comparison_table.to_csv(data_dir / "candidate_model_comparison.csv")
""",
)


def make_display_capture(cell_outputs: list) -> object:
    """Return a display callback bound to one notebook cell's output list."""

    def capture_display(value: object) -> None:
        data = {"text/plain": str(value)}
        if hasattr(value, "to_html"):
            data["text/html"] = value.to_html()
        cell_outputs.append(nbformat.v4.new_output("display_data", data=data))

    return capture_display


notebook = nbformat.v4.new_notebook(cells=cells)
notebook.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
notebook.metadata["language_info"] = {"name": "python"}
env = {"__name__": "__main__"}
count = 0
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    count += 1
    outputs = []
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(cell.source, f"<comparison cell {count}>", "exec"), env)
    env["display"] = make_display_capture(outputs)
    if stream.getvalue():
        outputs.insert(0, nbformat.v4.new_output("stream", name="stdout", text=stream.getvalue()))
    cell.execution_count = count
    cell.outputs = outputs
    print(f"Completed cell {count}", flush=True)
nbformat.validate(notebook)
output = root / "notebooks/11_Model_Comparison.ipynb"
nbformat.write(notebook, output)
print(env["comparison_table"].to_string(), flush=True)
print(f"Saved {output}", flush=True)
