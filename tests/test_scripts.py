"""Every script at least starts and explains itself.

The scripts are how the results are produced, so each one must at least import its dependencies,
parse its arguments and print usage. (Their full runs need the 1.6 GB data file and take minutes,
so they are exercised by hand and recorded in ``results/``.)
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


def test_there_are_scripts_to_check():
    assert len(SCRIPTS) >= 8


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_prints_usage_and_exits_cleanly(script):
    done = subprocess.run([sys.executable, str(script), "--help"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-500:]
    assert "usage:" in done.stdout.lower()
    assert "--" in done.stdout                         # at least one documented option


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_has_a_module_docstring_with_a_usage_example(script):
    text = script.read_text(encoding="utf-8")
    assert text.lstrip().startswith('"""'), f"{script.name} has no module docstring"
    assert "python scripts/" in text or "Usage" in text, f"{script.name} docstring has no usage example"
