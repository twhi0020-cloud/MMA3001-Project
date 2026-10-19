"""Rebuild the HTML documentation (``docs/``) and the test reports (``reports/``) in one command.

Run it after changing any code, so the documentation and reports stay in step with it. From the
repository root::

    python scripts/build_docs_and_reports.py              # documentation, then the full test suite with reports
    python scripts/build_docs_and_reports.py --skip-tests # documentation only (fast)

It writes ``docs/`` (pdoc, from the docstrings), ``reports/test_report.html`` (pytest-html),
``reports/junit.xml`` and ``reports/coverage.txt`` (pytest-cov), and removes this machine's
folder paths from them. The exit code is pytest's, so a failing test fails the script.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_FILES = ("test_report.html", "junit.xml", "coverage.txt")


def build_docs() -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, "-m", "pdoc", "occupancy_forecast", "-o", "docs", "--docformat", "google"],
                   cwd=ROOT, env=env, check=True)
    print("Wrote docs/")


def run_tests() -> int:
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--html=reports/test_report.html", "--self-contained-html",
         "--junitxml=reports/junit.xml", "--cov=occupancy_forecast", "--cov-report=term-missing"],
        cwd=ROOT, capture_output=True, text=True)
    lines = done.stdout.splitlines()
    summary = next((l for l in reversed(lines) if " passed" in l or " failed" in l or " error" in l), "").strip()
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("Name") and "Stmts" in l)
        end = next(i for i, l in enumerate(lines) if l.startswith("TOTAL"))
        table = "\n".join(lines[start:end + 1])
    except StopIteration:
        table = "(coverage table not available)"
    (reports / "coverage.txt").write_text(
        "Line coverage of src/occupancy_forecast by the test suite (pytest-cov).\n"
        f"Run: {summary}; Python {sys.version.split()[0]}.\n"
        "'Missing' lists the line numbers no test executes (mostly defensive branches).\n\n" + table + "\n",
        encoding="utf-8")
    for name in REPORT_FILES:                       # make the saved files portable
        p = reports / name
        if p.exists():
            text = p.read_text(encoding="utf-8")
            for form in (ROOT.as_posix() + "/", ROOT.as_posix(), str(ROOT) + os.sep, str(ROOT)):
                text = text.replace(form, "" if form.endswith(("/", os.sep)) else ".")
            p.write_text(text, encoding="utf-8")
    print(done.stdout[-600:])
    print("Wrote reports/test_report.html, reports/junit.xml, reports/coverage.txt")
    return done.returncode


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--skip-docs", action="store_true", help="do not regenerate docs/")
    p.add_argument("--skip-tests", action="store_true", help="do not run the tests or write reports/")
    args = p.parse_args(argv)
    if not args.skip_docs:
        build_docs()
    return 0 if args.skip_tests else run_tests()


if __name__ == "__main__":
    sys.exit(main())
