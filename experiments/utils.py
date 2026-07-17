# src/experiments/utils.py
from pathlib import Path

# Directory where this file is located: src/experiments
CURRENT_DIR = Path(__file__).resolve().parent

# Main root directory of the project: one level up (src) and another level up (root)
ROOT_DIR = CURRENT_DIR.parent

# Dynamic directory paths
DATA_DIR = ROOT_DIR / "data"
FIGURES_DIR = ROOT_DIR / "report" / "figures"


def get_data_path(filename: str) -> Path:
    """Returns the dynamic absolute path of the data file."""
    return DATA_DIR / filename


def get_figure_path(filename: str) -> Path:
    """Returns the dynamic absolute path of the figure file."""
    return FIGURES_DIR / filename