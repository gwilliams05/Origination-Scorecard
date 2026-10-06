"""Regression tests for the public synthetic-data generator."""

from pathlib import Path

import pandas as pd
import pytest

from origination_scorecard import GeneratorConfig, generate_project_data, write_project_data


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_accounts": 999}, "n_accounts"),
        ({"target_known_bad_rate": -0.01}, "target_known_bad_rate"),
        ({"target_inferred_rate": 1.0}, "target_inferred_rate"),
        (
            {"target_known_bad_rate": 0.6, "target_inferred_rate": 0.4},
            "must sum to less than 1",
        ),
        ({"broken_link_rate": 0.051}, "broken_link_rate"),
    ],
)
def test_invalid_configurations_are_rejected(
    overrides: dict[str, int | float], message: str
) -> None:
    config = GeneratorConfig(**overrides)

    with pytest.raises(ValueError, match=message):
        config.validate()


def test_generation_is_deterministic_and_has_expected_schema() -> None:
    config = GeneratorConfig(n_accounts=1_000, random_seed=12345)

    first = generate_project_data(config)
    second = generate_project_data(config)

    assert set(first) == {
        "applications",
        "performance_12m",
        "experian_consumer",
        "experian_consumer_dictionary",
    }
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])

    assert "master_account_key" not in first["applications"]
    assert "account_id" in first["applications"]
    assert len(first["applications"]) > 0
    assert len(first["performance_12m"].filter(regex=r"^missed_payment_m\d{2}$").columns) == 12


def test_write_project_data_creates_expected_csv_files(tmp_path: Path) -> None:
    outputs = generate_project_data(GeneratorConfig(n_accounts=1_000, random_seed=7))

    write_project_data(outputs, tmp_path)

    for name, expected in outputs.items():
        output_path = tmp_path / "data" / f"{name}.csv"
        assert output_path.is_file()
        actual = pd.read_csv(output_path)
        assert actual.shape == expected.shape
