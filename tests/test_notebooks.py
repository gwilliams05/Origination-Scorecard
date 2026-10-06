"""Structural quality checks for the complete notebook sequence."""

import re
from pathlib import Path

import nbformat
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIRECTORY = PROJECT_ROOT / "notebooks"
NUMBER_PATTERN = re.compile(r"^(\d{2})")


def notebook_paths() -> list[Path]:
    return sorted(NOTEBOOK_DIRECTORY.glob("*.ipynb"))


def test_notebook_sequence_is_complete_and_unique() -> None:
    numbers = [
        int(match.group(1))
        for path in notebook_paths()
        if (match := NUMBER_PATTERN.match(path.name)) is not None
    ]

    assert numbers == list(range(1, 18))


@pytest.mark.parametrize("path", notebook_paths(), ids=lambda path: path.stem)
def test_notebook_is_valid_and_standardised(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)

    assert notebook.cells, "notebook must contain at least one cell"
    assert notebook.cells[0].cell_type == "markdown"
    assert notebook.cells[0].source.lstrip().startswith("# ")
    assert all(cell.source.strip() for cell in notebook.cells)
    assert notebook.metadata.kernelspec.name == "python3"
    assert notebook.metadata.language_info.name == "python"
