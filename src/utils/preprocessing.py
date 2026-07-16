# Preprocessing utilities – notebook execution helpers

"""Utilities to run Jupyter notebooks programmatically.

Requires: nbformat, nbconvert
Usage:
    from src.utils.preprocessing import run_notebooks
    run_notebooks([
        "notebooks/adult_preprocessing.ipynb",
        "notebooks/covtype_preprocessing.ipynb",
        "notebooks/wdbc_preprocessing.ipynb",
    ])
"""

from pathlib import Path
from typing import Iterable, List

import sys
import asyncio

# On Windows, use the selector event loop for zmq compatibility to avoid RuntimeWarning
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        # Older Python or unexpected environment — ignore and continue
        pass

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run_notebook(path: str, timeout: int = 600, kernel_name: str = "python3", output_path: str | None = None) -> str:
    """Execute a single notebook and write the executed notebook to output_path (or overwrite).

    Args:
        path: Path to the input notebook (.ipynb).
        timeout: Cell execution timeout in seconds.
        kernel_name: Kernel to use (e.g., "python3").
        output_path: If provided, write executed notebook to this path. If None, overwrite the input file.

    Returns:
        The path to the executed notebook as a string.
    """
    nb_path = Path(path)
    if output_path is None:
        output_path = nb_path
    else:
        output_path = Path(output_path)

    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook not found: {nb_path}")

    with nb_path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)
    ep.preprocess(nb, {"metadata": {"path": str(nb_path.parent)}})

    with output_path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    return str(output_path)


def run_notebooks(paths: Iterable[str], timeout: int = 600, kernel_name: str = "python3", overwrite: bool = True) -> List[str]:
    """Execute multiple notebooks.

    Args:
        paths: Iterable of notebook file paths to execute.
        timeout: Per-notebook cell execution timeout.
        kernel_name: Kernel to use for execution.
        overwrite: If True, overwrite each input notebook with its executed version. If False, save executed versions with "_executed" suffix.

    Returns:
        List of paths to executed notebooks.
    """
    executed: List[str] = []
    for p in paths:
        p_path = Path(p)
        if not p_path.exists():
            raise FileNotFoundError(f"Notebook not found: {p}")

        if overwrite:
            out = p_path
        else:
            out = p_path.with_name(p_path.stem + "_executed.ipynb")

        executed_path = run_notebook(str(p_path), timeout=timeout, kernel_name=kernel_name, output_path=str(out))
        executed.append(executed_path)

    return executed


if __name__ == "__main__":
    """Run the three preprocessing notebooks when the module is executed as a script.

    Default notebook directory: project_root/notebooks
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run preprocessing notebooks")
    parser.add_argument("--notebooks-dir", type=str, default=None, help="Directory containing notebooks (default: project_root/notebooks)")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite notebooks; save executed copies with _executed suffix")
    parser.add_argument("--timeout", type=int, default=600, help="Per-cell execution timeout in seconds")
    parser.add_argument("--kernel", type=str, default="python3", help="Kernel name to use for execution")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    notebooks_dir = Path(args.notebooks_dir) if args.notebooks_dir else project_root / "src/utils"

    notebook_names = [
        "adult_preprocessing.ipynb",
        "covtype_preprocessing.ipynb",
        "wdbc_preprocessing.ipynb",
    ]

    notebook_paths = [str(notebooks_dir / name) for name in notebook_names]

    try:
        executed = run_notebooks(notebook_paths, timeout=args.timeout, kernel_name=args.kernel, overwrite=not args.no_overwrite)
        print("Executed notebooks:")
        for p in executed:
            print(" -", p)
    except Exception as exc:  # graceful failure reporting
        print("Error executing notebooks:", exc, file=sys.stderr)
        sys.exit(1)
