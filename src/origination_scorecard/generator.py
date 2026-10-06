"""Deterministic synthetic data generator for a credit-risk teaching project.

All apparent personal data produced here is fabricated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GeneratorConfig:
    """Validated settings for a reproducible synthetic portfolio."""

    n_accounts: int = 10_000
    random_seed: int = 20_260_907
    target_known_bad_rate: float = 0.04
    target_inferred_rate: float = 0.02
    broken_link_rate: float = 0.01

    def validate(self) -> None:
        """Raise ``ValueError`` when settings cannot form a useful portfolio."""

        if self.n_accounts < 1_000:
            raise ValueError("n_accounts must be at least 1,000 for a useful teaching sample")
        if not 0 <= self.target_known_bad_rate < 1:
            raise ValueError("target_known_bad_rate must be between 0 and 1")
        if not 0 <= self.target_inferred_rate < 1:
            raise ValueError("target_inferred_rate must be between 0 and 1")
        if self.target_known_bad_rate + self.target_inferred_rate >= 1:
            raise ValueError("bad and inferred rates must sum to less than 1")
        if not 0 <= self.broken_link_rate <= 0.05:
            raise ValueError("broken_link_rate must be between 0 and 0.05")


FIRST_NAMES = np.array(
    [
        "Amelia",
        "Oliver",
        "Isla",
        "George",
        "Ava",
        "Noah",
        "Freya",
        "Leo",
        "Maya",
        "Theo",
        "Sofia",
        "Adam",
        "Priya",
        "Daniel",
        "Aisha",
        "Jack",
    ]
)
LAST_NAMES = np.array(
    [
        "Bennett",
        "Clarke",
        "Davies",
        "Evans",
        "Foster",
        "Green",
        "Hughes",
        "Jones",
        "Khan",
        "Lewis",
        "Morgan",
        "Patel",
        "Roberts",
        "Shah",
        "Taylor",
        "Wilson",
    ]
)
STREETS = np.array(
    [
        "Oak Road",
        "Station Lane",
        "Meadow Close",
        "Church Street",
        "Mill View",
        "Kingfisher Way",
        "Park Avenue",
        "Willow Drive",
        "Victoria Road",
        "Highfield Gardens",
    ]
)
POST_AREAS = np.array(
    ["B", "BS", "CF", "EH", "G", "L", "LS", "M", "NE", "NG", "NR", "OX", "PL", "RG", "S", "SO"]
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _weighted_sample(rng: np.random.Generator, weights: np.ndarray, size: int) -> np.ndarray:
    probabilities = np.maximum(weights, 1e-12)
    probabilities = probabilities / probabilities.sum()
    return rng.choice(len(weights), size=size, replace=False, p=probabilities)


def _make_applications(config: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = config.n_accounts
    application_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 730, n), unit="D"
    )
    age = np.clip(np.rint(rng.normal(42, 12, n)), 18, 75).astype(int)
    dob = application_date - pd.to_timedelta(
        (age * 365.2425 + rng.integers(0, 365, n)).astype(int), unit="D"
    )
    employment_status = rng.choice(
        ["employed", "self_employed", "part_time", "retired", "unemployed"],
        n,
        p=[0.64, 0.12, 0.10, 0.08, 0.06],
    )
    residential_status = rng.choice(
        [
            "homeowner_mortgage",
            "homeowner_outright",
            "private_tenant",
            "social_tenant",
            "living_with_family",
        ],
        n,
        p=[0.36, 0.13, 0.29, 0.12, 0.10],
    )
    annual_income = np.clip(rng.lognormal(np.log(32_000), 0.48, n), 8_000, 180_000).round(-2)
    unemployed = employment_status == "unemployed"
    annual_income[unemployed] = rng.uniform(8_000, 18_000, unemployed.sum()).round(-2)
    monthly_net = (annual_income / 12 * np.clip(rng.normal(0.76, 0.025, n), 0.66, 0.82)).round(2)
    housing_cost = np.where(
        residential_status == "homeowner_outright",
        rng.uniform(0, 180, n),
        np.clip(rng.normal(760, 290, n), 180, 2_400),
    ).round(2)
    committed = np.clip(rng.gamma(2.1, 150, n), 20, 2_000).round(2)
    disposable = (monthly_net - housing_cost - committed).round(2)
    debt_to_income = np.clip((committed * 12) / annual_income, 0, 1.5).round(4)
    util = np.clip(rng.beta(2.0, 3.0, n) + rng.normal(0, 0.05, n), 0, 1).round(4)
    searches = np.clip(rng.poisson(1.25, n), 0, 9)
    prior_defaults = rng.binomial(2, 0.055, n)
    prior_arrears = np.clip(rng.poisson(0.24, n) + prior_defaults, 0, 8)
    employment_months = (
        np.clip((rng.gamma(2.3, 28, n) - unemployed * 50), 0, 480).round().astype(int)
    )
    address_months = np.clip(rng.gamma(2.2, 30, n), 0, 600).round().astype(int)
    credit_history = np.clip((age - 18) * 12 * rng.uniform(0.3, 1, n), 0, 600).round().astype(int)
    area = rng.choice(POST_AREAS, n)
    postcode = np.array(
        [
            f"{a}{num} {digit}{letters}"
            for a, num, digit, letters in zip(
                area,
                rng.integers(1, 30, n),
                rng.integers(1, 10, n),
                rng.choice(list("ABDEFGHJLNPQRSTUVWXY"), (n, 2)).view("U2").reshape(-1),
                strict=True,
            )
        ]
    )

    return pd.DataFrame(
        {
            "master_account_key": [f"MASTER-{i:07d}" for i in range(1, n + 1)],
            "clean_account_id": [f"ACC-{10000000 + i}" for i in range(n)],
            "customer_id": [f"CUS-{20000000 + i}" for i in range(n)],
            "customer_name": rng.choice(FIRST_NAMES, n) + " " + rng.choice(LAST_NAMES, n),
            "address_line_1": rng.integers(1, 250, n).astype(str) + " " + rng.choice(STREETS, n),
            "postcode": postcode,
            "date_of_birth": pd.to_datetime(dob).date.astype(str),
            "age_at_application": age,
            "application_date": pd.to_datetime(application_date).date.astype(str),
            "employment_status": employment_status,
            "employment_tenure_months": employment_months,
            "occupation_group": rng.choice(
                [
                    "professional",
                    "managerial",
                    "skilled",
                    "administrative",
                    "service",
                    "manual",
                    "not_working",
                ],
                n,
            ),
            "residential_status": residential_status,
            "time_at_address_months": address_months,
            "household_size": np.clip(rng.poisson(1.5, n) + 1, 1, 7),
            "dependants": np.clip(rng.poisson(0.7, n), 0, 5),
            "annual_gross_income": annual_income,
            "estimated_monthly_net_income": monthly_net,
            "committed_monthly_outgoings": committed,
            "rent_or_mortgage": housing_cost,
            "disposable_income": disposable,
            "debt_to_income_ratio": debt_to_income,
            "requested_loan_amount": (
                np.clip(rng.lognormal(np.log(7_000), 0.65, n), 500, 30_000) / 100
            ).round()
            * 100,
            "loan_purpose": rng.choice(
                [
                    "debt_consolidation",
                    "car",
                    "home_improvement",
                    "major_purchase",
                    "education",
                    "other",
                ],
                n,
                p=[0.33, 0.25, 0.18, 0.12, 0.05, 0.07],
            ),
            "bank_account_tenure_months": np.clip(rng.gamma(2.4, 31, n), 0, 600)
            .round()
            .astype(int),
            "income_verification_result": rng.choice(
                ["verified", "partially_verified", "not_verified"], n, p=[0.77, 0.16, 0.07]
            ),
            "application_channel": rng.choice(
                ["web", "mobile", "branch", "telephone"], n, p=[0.45, 0.35, 0.12, 0.08]
            ),
            "application_hour": rng.integers(0, 24, n),
            "recent_application_count": searches,
            "prior_customer_flag": rng.binomial(1, 0.27, n),
            "device_verification_result": rng.choice(
                ["pass", "refer", "fail"], n, p=[0.91, 0.07, 0.02]
            ),
            "contact_verification_result": rng.choice(
                ["pass", "refer", "fail"], n, p=[0.93, 0.055, 0.015]
            ),
            "credit_history_months": credit_history,
            "active_credit_accounts": np.clip(rng.poisson(3.2, n), 0, 14),
            "revolving_utilisation": util,
            "recent_credit_searches": searches,
            "historic_defaults": prior_defaults,
            "historic_arrears": prior_arrears,
            "electoral_roll_match": rng.choice(
                ["full", "partial", "none"], n, p=[0.79, 0.14, 0.07]
            ),
        }
    )


def _make_performance(
    app: pd.DataFrame, config: GeneratorConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    instability = (app["employment_tenure_months"].to_numpy() < 12).astype(float) + (
        app["time_at_address_months"].to_numpy() < 12
    )
    risk_score = (
        1.15 * app["revolving_utilisation"].to_numpy()
        + 0.22 * app["recent_credit_searches"].to_numpy()
        + 0.75 * app["historic_defaults"].to_numpy()
        + 0.28 * app["historic_arrears"].to_numpy()
        + 0.48 * instability
        + 1.15 * (app["disposable_income"].to_numpy() < 150)
        + 0.65 * (app["debt_to_income_ratio"].to_numpy() > 0.38)
        + rng.normal(0, 0.85, len(app))
    )
    n_bad = round(config.n_accounts * config.target_known_bad_rate)
    n_inferred = round(config.n_accounts * config.target_inferred_rate)
    bad_idx = _weighted_sample(rng, _sigmoid(risk_score - 2.4), n_bad)
    remaining = np.setdiff1d(np.arange(config.n_accounts), bad_idx)
    inferred_rel = _weighted_sample(rng, _sigmoid(risk_score[remaining] - 2.0), n_inferred)
    inferred_idx = remaining[inferred_rel]
    outcome = np.full(config.n_accounts, "known_good", dtype=object)
    outcome[inferred_idx] = "inferred"
    outcome[bad_idx] = "known_bad"
    missed = np.zeros((config.n_accounts, 12), dtype=int)
    for idx in inferred_idx:
        months = rng.choice(12, size=int(rng.integers(1, 3)), replace=False)
        missed[idx, months] = 1
    for idx in bad_idx:
        count = int(np.clip(3 + rng.poisson(1.6), 3, 9))
        missed[idx, rng.choice(12, size=count, replace=False)] = 1
    perf = pd.DataFrame({"clean_account_id": app["clean_account_id"]})
    for month in range(12):
        perf[f"missed_payment_m{month + 1:02d}"] = missed[:, month]
    truth = pd.DataFrame(
        {
            "master_account_key": app["master_account_key"],
            "clean_account_id": app["clean_account_id"],
            "true_outcome": outcome,
            "total_missed_payments": missed.sum(axis=1),
            "latent_risk_score": risk_score.round(6),
        }
    )
    return perf, truth


def _make_experian_consumer(app: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Create a compact Delphi Select-style consumer summary.

    Field names and meanings follow the Consumer mock/schema in RT-Howe/corezoid_poc.
    Values and distributions are independently synthetic teaching assumptions, not
    Experian data or calibrated Experian distributions.
    """
    n = len(app)
    defaults = np.clip(app["historic_defaults"].to_numpy(), 0, 9).astype(int)
    arrears = np.clip(app["historic_arrears"].to_numpy(), 0, 9).astype(int)
    searches_3m = np.clip(app["recent_credit_searches"].to_numpy(), 0, 9).astype(int)
    searches_6m = np.clip(searches_3m + rng.poisson(0.8, n), 0, 9).astype(int)
    active = np.clip(app["active_credit_accounts"].to_numpy(), 0, 9).astype(int)
    status_one = np.minimum(
        active,
        rng.binomial(
            np.maximum(active, 1),
            np.clip(0.02 + 0.22 * app["revolving_utilisation"].to_numpy(), 0, 0.45),
        ),
    )
    severe_6m = np.clip(defaults + rng.binomial(2, np.clip(arrears / 12, 0, 0.65)), 0, 9).astype(
        int
    )
    worst_status = np.select(
        [defaults > 0, severe_6m > 0, arrears > 0, active > 0],
        ["D", "3", "1", "0"],
        default="N",
    )
    public_records = np.clip(defaults + rng.binomial(1, 0.012, n), 0, 9).astype(int)
    ccj_value_band = np.where(
        public_records > 0, np.clip(np.rint(rng.lognormal(2.0, 0.8, n)), 1, 999), 0
    ).astype(int)
    bankruptcy_iva = np.where((defaults >= 2) & (rng.random(n) < 0.18), "Y", "N")
    raw_risk = (
        55 * defaults
        + 23 * arrears
        + 16 * searches_3m
        + 165 * app["revolving_utilisation"].to_numpy()
        + 35 * (app["electoral_roll_match"].to_numpy() == "none")
        + rng.normal(0, 42, n)
    )
    delphi_score = np.clip(np.rint(925 - raw_risk), 100, 999).astype(int)

    return pd.DataFrame(
        {
            "account_id": app["clean_account_id"],
            "DelphiScore": delphi_score,
            "E1A04": defaults,
            "E1A09": np.minimum(arrears, 9),
            "E1B01": np.minimum(searches_3m, active),
            "E1B07": worst_status,
            "E1B09": active,
            "E1B12": status_one.astype(int),
            "E1B13": severe_6m,
            "E1E01": searches_3m,
            "E1E02": searches_6m,
            "E1A01": public_records,
            "E1A02": ccj_value_band,
            "EA1C01": bankruptcy_iva,
            "EA4Q01": np.where(app["electoral_roll_match"].to_numpy() == "none", "T", "N"),
        }
    )


def _experian_consumer_dictionary() -> pd.DataFrame:
    source = "RT-Howe/corezoid_poc Delphi Select v2 Consumer schema, commit b03759a"
    definitions = [
        ("account_id", "Project linkage identifier; not an Experian field", "string"),
        ("DelphiScore", "Delphi score", "integer"),
        (
            "E1A04",
            "Number of default CAIS accounts excluding mail order accounts (SP)",
            "integer code: -1 to 9",
        ),
        (
            "E1A09",
            "Number of delinquent CAIS accounts excluding mail order accounts (SP)",
            "integer code: -1 to 9",
        ),
        (
            "E1B01",
            "Number of active CAIS accounts opened in the last 3 months (SP)",
            "integer code: -1 to 9",
        ),
        (
            "E1B07",
            "Worst status in the last 6 months of all active CAIS accounts (SP)",
            "status code",
        ),
        ("E1B09", "Number of active CAIS accounts (SP)", "integer code: -1 to 9"),
        (
            "E1B12",
            "Number of active CAIS accounts with current status 1 (SP)",
            "integer code: -1 to 9",
        ),
        (
            "E1B13",
            "Number of CAIS status 3 or worse events in the last 6 months (SP)",
            "integer code: -1 to 9",
        ),
        ("E1E01", "Number of search records in the last 3 months (SP)", "integer code: -1 to 9"),
        ("E1E02", "Number of search records in the last 6 months (SP)", "integer code: -1 to 9"),
        ("E1A01", "Number of public information records (SP)", "integer code: -1 to 9"),
        ("E1A02", "Total value of CCJs banded in £100 intervals (SP)", "integer code: -1 to 999"),
        ("EA1C01", "Bankruptcy or IVA detected (SP)", "T/Y/N indicator"),
        ("EA4Q01", "Voters Roll notice-of-correction indicator (All)", "T/Y/N indicator"),
    ]
    result = pd.DataFrame(definitions, columns=["field", "schema_meaning", "schema_type_or_range"])
    result["source"] = source
    result["teaching_note"] = (
        "Schema-aligned field; generated values are synthetic and are not an Experian distribution"
    )
    return result


def _inject_linkage_defects(
    app: pd.DataFrame, perf: pd.DataFrame, config: GeneratorConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    application_output = (
        app.drop(columns=["master_account_key"])
        .rename(columns={"clean_account_id": "account_id"})
        .copy()
    )
    performance_output = perf.rename(columns={"clean_account_id": "account_id"}).copy()
    defect_n = max(1, round(config.n_accounts * config.broken_link_rate))
    pool = rng.permutation(config.n_accounts)
    chunks = np.array_split(pool[:defect_n], 4)

    for idx in chunks[0]:
        original = application_output.at[idx, "account_id"]
        application_output.at[idx, "account_id"] = original.replace("ACC-", "acc ")
    for idx in chunks[1]:
        performance_output.at[idx, "account_id"] = None
    for idx in chunks[2]:
        original = performance_output.at[idx, "account_id"]
        performance_output.at[idx, "account_id"] = original[:-2] + "XX"
    drop_app_idx = set(int(x) for x in chunks[3])
    application_output = application_output.drop(index=list(drop_app_idx)).reset_index(drop=True)

    duplicate_count = max(1, defect_n // 8)
    duplicate_idx = rng.choice(len(application_output), duplicate_count, replace=False)
    duplicates = application_output.iloc[duplicate_idx].copy()
    application_output = pd.concat([application_output, duplicates], ignore_index=True)

    orphan_count = max(1, defect_n // 8)
    orphans = performance_output.iloc[:orphan_count].copy()
    orphans["account_id"] = [f"ACC-ORPHAN-{i:04d}" for i in range(orphan_count)]
    performance_output = pd.concat([performance_output, orphans], ignore_index=True)

    return application_output, performance_output


def generate_project_data(config: GeneratorConfig) -> dict[str, pd.DataFrame]:
    """Generate all project datasets in memory without writing to disk."""
    config.validate()
    rng = np.random.default_rng(config.random_seed)
    app = _make_applications(config, rng)
    perf, _ = _make_performance(app, config, rng)
    experian_consumer = _make_experian_consumer(app, rng)
    application_output, performance_output = _inject_linkage_defects(app, perf, config, rng)
    return {
        "applications": application_output,
        "performance_12m": performance_output,
        "experian_consumer": experian_consumer,
        "experian_consumer_dictionary": _experian_consumer_dictionary(),
    }


def write_project_data(outputs: Mapping[str, pd.DataFrame], project_root: Path) -> None:
    """Write generated datasets beneath ``project_root / 'data'`` as CSV files."""

    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    outputs["applications"].to_csv(data_dir / "applications.csv", index=False)
    outputs["performance_12m"].to_csv(data_dir / "performance_12m.csv", index=False)
    outputs["experian_consumer"].to_csv(data_dir / "experian_consumer.csv", index=False)
    outputs["experian_consumer_dictionary"].to_csv(
        data_dir / "experian_consumer_dictionary.csv", index=False
    )
