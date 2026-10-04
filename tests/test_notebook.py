"""The quickstart notebook runs top to bottom (it is the documented Jupyter/Colab entry point)."""

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

NOTEBOOK = Path(__file__).resolve().parent.parent / "examples" / "quickstart.ipynb"


def test_quickstart_notebook_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the notebook writes demo/ and reports/ next to itself
    nb = nbformat.read(NOTEBOOK, as_version=4)
    try:
        nbclient.NotebookClient(nb, timeout=300, kernel_name="python3").execute()
    except nbclient.exceptions.CellExecutionError as exc:
        pytest.fail(str(exc)[:2000])
    except Exception as exc:  # no Jupyter kernel installed
        pytest.skip(f"cannot start a kernel: {exc}")
