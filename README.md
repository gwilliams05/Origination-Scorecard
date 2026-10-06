# Origination Scorecard

An educational Python project for developing and comparing logistic-regression
probability-of-default (PD) scorecards on synthetic UK consumer-credit data.

> [!IMPORTANT]
> The generated names, addresses, account details, and credit attributes are fictional.
> This repository is for teaching and experimentation, not production lending decisions.

## Requirements

- Python 3.12–3.14
- [`uv`](https://docs.astral.sh/uv/) for dependency and environment management

Create or refresh the local environment, including the development tools:

```powershell
uv sync --dev
```

## Common workflows

Generate the default synthetic datasets and the two starter notebooks:

```powershell
uv run python scripts/build_project.py
```

This command overwrites `notebooks/01_generate_data.ipynb`,
`notebooks/02_scorecard_development.ipynb`, and its generated CSV outputs. Use the
command-line options shown by `--help` to change the sample size, random seed, event
rates, or linkage-defect rate.

Start JupyterLab:

```powershell
uv run jupyter lab
```

Run all repository quality checks:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Apply safe lint fixes and formatting before committing:

```powershell
uv run ruff check . --fix
uv run ruff format .
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/origination_scorecard/` | Reusable synthetic-data generation code |
| `scripts/` | Command-line analysis and notebook-building utilities |
| `notebooks/` | Step-by-step scorecard development and policy analysis |
| `data/` | Synthetic source, modelling, and derived datasets |
| `tests/` | Fast regression tests for the reusable Python code |

## Coding standards

- Target Python 3.12 syntax and add type hints to reusable functions.
- Keep reusable logic under `src/`; reserve notebooks for exploration and explanation.
- Estimate transformations on training data only, then apply them unchanged to validation
  data to avoid leakage.
- Use deterministic random seeds for generated data and model experiments.
- Never use fabricated personal identifiers as model predictors.
- Run Ruff and pytest before committing. Generated datasets and notebooks are excluded
  from linting so checks remain focused on maintained Python source.
