"""Build notebooks and the default synthetic CSV outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from origination_scorecard.generator import GeneratorConfig, generate_project_data, write_project_data


def _markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


ROOT_FINDER = """def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / 'data').is_dir() and (candidate / 'src').is_dir():
            return candidate
    raise FileNotFoundError('Open this notebook from the origination-scorecard-dev workspace')

PROJECT_ROOT = find_project_root(Path.cwd().resolve())"""


def _write_notebook(path: Path, title: str, cells: list[dict]) -> None:
    notebook = {
        "cells": [_markdown(f"# {title}\n\n**Synthetic teaching data only.** Do not use PII as model predictors.")] + cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


def build_notebooks() -> None:
    generator_source = (PROJECT_ROOT / "src" / "origination_scorecard" / "generator.py").read_text(encoding="utf-8")
    generator_cells = [
        _markdown("## How to use this notebook\n\nRun cells in order with **Shift+Enter**. Change the settings below to create a different, reproducible portfolio."),
        _code("from pathlib import Path\n\n" + ROOT_FINDER),
        _code("n_accounts = 10_000\nrandom_seed = 20_260_907\ntarget_known_bad_rate = 0.04\ntarget_inferred_rate = 0.02\nbroken_link_rate = 0.01"),
        _markdown("## Data-generation code\n\nThe complete generator is included below so its assumptions and transformations can be inspected and changed within the notebook. Run this cell before building the datasets."),
        _code(generator_source),
        _code("config = GeneratorConfig(n_accounts=n_accounts, random_seed=random_seed, target_known_bad_rate=target_known_bad_rate, target_inferred_rate=target_inferred_rate, broken_link_rate=broken_link_rate)\noutputs = generate_project_data(config)\nwrite_project_data(outputs, PROJECT_ROOT)\n{name: frame.shape for name, frame in outputs.items()}"),
        _markdown("## Check the generated files\n\nThe same run creates applications, 12-month performance and the Experian Consumer-style source."),
        _code("generated_files = [\n    PROJECT_ROOT / 'data' / 'applications.csv',\n    PROJECT_ROOT / 'data' / 'performance_12m.csv',\n    PROJECT_ROOT / 'data' / 'experian_consumer.csv',\n    PROJECT_ROOT / 'data' / 'experian_consumer_dictionary.csv',\n]\n[(path.relative_to(PROJECT_ROOT).as_posix(), path.exists(), round(path.stat().st_size / 1_000_000, 2)) for path in generated_files]"),
        _code("outputs['experian_consumer'].head()"),
        _markdown("## Experian Consumer source\n\nThe build also writes `data/experian_consumer.csv` and its data dictionary. Field names and meanings follow the Consumer schema in `RT-Howe/corezoid_poc`; values remain entirely synthetic teaching assumptions."),
    ]
    learner_cells = [
        _markdown("## Notebook basics\n\nA notebook contains text and executable code cells. Run one cell at a time with **Shift+Enter**. Restart the kernel and run all cells when you want to confirm that the work is reproducible."),
        _code("from pathlib import Path\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import roc_auc_score\n\n" + ROOT_FINDER + "\nDATA_DIR = PROJECT_ROOT / 'data'"),
        _code("applications = pd.read_csv(DATA_DIR / 'applications.csv')\nperformance = pd.read_csv(DATA_DIR / 'performance_12m.csv')\nexperian_consumer = pd.read_csv(DATA_DIR / 'experian_consumer.csv')\napplications.shape, performance.shape, experian_consumer.shape"),
        _markdown("## 1. Inspect and profile\n\nExamine column types, missing values, duplicate rows and plausible ranges. Record anything that could affect a join or a model."),
        _code("# TODO: inspect both datasets\napplications.head()"),
        _markdown("## 2. Repair the linkage\n\nDevelop a defensible normalisation rule for account IDs. Quantify duplicates and unmatched rows across the application, performance and Experian Consumer-style sources. Do not silently fuzzy-match uncertain identifiers."),
        _code("# TODO: create normalised join keys and produce a documented one-row-per-account join"),
        _markdown("## 3. Define outcomes\n\nSum the twelve binary monthly fields. Define 0 as known good, 1–2 as inferred and 3+ as known bad. Check the observed rates after linkage."),
        _code("# TODO: derive total missed payments and the three outcome groups"),
        _markdown("## 4. Development and out-of-time samples\n\nExclude inferred accounts from initial development. Use earlier applications for development and later applications for out-of-time validation."),
        _code("# TODO: choose and justify a date boundary"),
        _markdown("## 5. Traditional scorecard\n\nExclude PII and post-outcome information. Bin candidate variables, calculate Weight of Evidence and Information Value, fit logistic regression, and inspect signs and stability."),
        _code("# TODO: implement coarse classing, WoE/IV and logistic regression"),
        _markdown("## 6. Validation and inferred population\n\nAssess discrimination and calibration out of time. Scale model output into scorecard points, then score the inferred population and discuss the limitations."),
        _code("# TODO: validate, create points and examine inferred cases"),
        _markdown("## Discussion\n\nHow does known-good/known-bad modelling differ from analysing an inferred population? What assumptions would be required for reject inference?"),
    ]
    _write_notebook(PROJECT_ROOT / "notebooks" / "01_generate_data.ipynb", "Generate the synthetic portfolio", generator_cells)
    _write_notebook(PROJECT_ROOT / "notebooks" / "02_scorecard_development.ipynb", "Origination scorecard development", learner_cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-accounts", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_907)
    parser.add_argument("--bad-rate", type=float, default=0.04)
    parser.add_argument("--inferred-rate", type=float, default=0.02)
    parser.add_argument("--broken-link-rate", type=float, default=0.01)
    args = parser.parse_args()
    config = GeneratorConfig(args.n_accounts, args.seed, args.bad_rate, args.inferred_rate, args.broken_link_rate)
    write_project_data(generate_project_data(config), PROJECT_ROOT)
    build_notebooks()
    print(f"Built synthetic project at {PROJECT_ROOT}")


if __name__ == "__main__":
    main()
