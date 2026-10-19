"""Guards on the committed notebooks: valid, fully executed, error-free, portable and not bloated.

The notebooks are the project's visual evidence, and they are committed *with* their outputs so
they render on GitHub. These tests stop a notebook being committed half-run, with a traceback in
it, with a path from someone's machine in it, or with an unreasonable size.
"""

import re
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))
pytestmark = pytest.mark.skipif(not NOTEBOOKS, reason="no notebooks in the repository")

MACHINE_PATHS = re.compile(r"(/home/|/tmp/|/root/|/mnt/|/Users/|[A-Za-z]:\\+Users)")
MAX_BYTES = 2_000_000


def _read(path):
    return nbformat.read(path, as_version=4)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_and_every_code_cell_was_executed(path):
    nb = _read(path)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert code, "notebook has no code cells"
    assert all(c.execution_count is not None for c in code), "a code cell has not been run"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_error_outputs(path):
    errors = [o for c in _read(path).cells if c.cell_type == "code" for o in c.get("outputs", [])
              if o.output_type == "error"]
    assert not errors, f"error output in {path.name}: {errors[0].get('ename')}: {errors[0].get('evalue')}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_contains_no_machine_specific_paths(path):
    hit = MACHINE_PATHS.search(path.read_text(encoding="utf-8"))
    assert hit is None, f"{path.name} contains a machine-specific path near {hit.group(0)!r}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_uses_the_library_and_is_not_bloated(path):
    nb = _read(path)
    source = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "occupancy_forecast" in source, "notebook should call the tested library, not re-implement it"
    assert path.stat().st_size < MAX_BYTES, f"{path.name} is {path.stat().st_size / 1e6:.1f} MB; clear or shrink outputs"
