# Test reports

| File | What it is |
|---|---|
| `test_report.html` | Every test and its result (pytest-html). Download it and open it in a browser |
| `junit.xml` | The same results in JUnit XML, for tools that read it |
| `coverage.txt` | Which lines of `src/occupancy_forecast` the tests execute, per module (pytest-cov) |

They were produced in a fresh virtual environment built from `requirements-lock.txt` (Python 3.12). To rebuild
them, and the HTML documentation in `docs/`, from the repository root:

```
pip install -r requirements-lock.txt
python scripts/build_docs_and_reports.py
```

The tests cover the library, the notebooks (valid, executed, error-free), the documentation (docstrings in step
with the code, HTML in step with the docstrings) and the scripts (each starts and prints usage). The full runs of
the scripts need the raw data file and take minutes; their outputs are recorded in `results/`.
