"""Fail if any code cell in the notebook has not been executed.

The notebook is delivered with every cell already run, so its outputs can be read without
rerunning it. This check runs in CI, so a cell that was edited and not rerun cannot reach main.

Usage: python scripts/check_notebook_executed.py vesuvius_burn_severity.ipynb
"""

import sys

import nbformat


def main(path: str) -> int:
    nb = nbformat.read(path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code" and c.source.strip()]
    not_run = [i for i, c in enumerate(nb.cells) if c in code_cells and c.execution_count is None]
    if not_run:
        print(f"{path}: code cells never executed, by index: {not_run}")
        return 1
    print(f"{path}: all {len(code_cells)} code cells have been executed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
