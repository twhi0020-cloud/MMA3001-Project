# Short-horizon room headcount forecasting

**MMA3001 Numerical Methods and Machine Learning, individual project (2026). Author: Toby White.**

This project forecasts how many people will be in a room 15, 30 or 60 minutes ahead from building occupancy sensor readings, so that building systems such as heating and cooling can respond before people arrive. The data are Monash Smart Infrastructure occupancy sensors: five rooms, about 9.1 million readings, November 2023 to April 2026.

**In brief.** A decision tree forecasts headcount better than "the same as now" at every horizon (9.2% lower RMSE at 30 minutes, on a test set opened once), but the improvement is modest in absolute terms, and a simpler spline-based model with one fitted parameter is statistically indistinguishable from the tree. Much of the work went into the data: the sensors are only partly linked to rooms, about half the timeline has no readings, and the choices made about that were tested to see whether they matter.

**Contents:** [1. Engineering problem](#1-engineering-problem) | [2. Inputs and outputs](#2-inputs-and-outputs) | [3. Approach](#3-approach) | [4. Results](#4-results) | [5. Known data issues](#5-known-data-issues) | [6. Limitations](#6-limitations) | [7. Getting started](#7-getting-started) | [8. Repository layout](#8-repository-layout) | [9. Testing, documentation and reproducibility](#9-testing-documentation-and-reproducibility) | [10. AI use](#10-ai-use) | [11. Licence and data](#11-licence-and-data)

## 1. Engineering problem

**Context.** Heating, ventilation and air-conditioning (HVAC) systems take time to bring a room to a comfortable condition. They are usually driven by fixed schedules or by the occupancy they can see right now, so they either react after people have already arrived or condition empty rooms. A forecast of the next half hour's occupancy lets a controller act ahead of demand.

**Problem.** Given the recent occupancy history of a room, predict its headcount `h` minutes ahead, for `h` = 15, 30 and 60. The sensor already reports the headcount *now*; the task is to predict what it *will be*. Predicting the current value would be a lookup, not a forecast, so every method is compared against the "nothing changes" forecast (persistence).

**Intended use.** A building-management process calls `forecast_headcount()` every few minutes with the last day of raw sensor readings and receives a forecast per room.

**Scope.** Five rooms in one building, reported by one sensor hub. Headcount is whatever the sensor reports (there is no independent count). The environmental sensors in the dataset are not used for forecasting, because they could not be linked to the same rooms (section 5). Results describe these rooms and these months.

## 2. Inputs and outputs

### The forecast function

`occupancy_forecast.forecast.forecast_headcount(readings, model, as_of=None)`

**Input: `readings`**, one row per sensor reading, in the format of the sensor export.

| Column | Meaning | Unit and format | Valid values |
|---|---|---|---|
| `floorspaceid` | Room identifier | text | any non-missing text |
| `headcount` | People counted by the sensor | people (a number) | zero or more; rows with a missing or non-numeric value are dropped |
| `collecteddate` | Time of the reading | text or datetime **with a UTC offset**, e.g. `2024-03-01 09:00:02+11` | must carry an offset |

`model` is a saved model bundle (written by `scripts/train_final.py`). `as_of` is the time of the last complete 5-minute bin; by default it is the latest reading's bin, and readings after it are ignored. At least about 65 minutes of history is needed.

**Output**, one row per room:

| Column | Meaning |
|---|---|
| `floorspaceid` | Room identifier |
| `as_of` | Start of the last complete 5-minute bin (Melbourne local time) |
| `forecast_for` | Start of the 5-minute bin being predicted: `as_of` plus the horizon |
| `predicted_headcount` | Forecast headcount, in people; never negative; missing if no forecast was possible |
| `current_headcount` | The latest headcount the forecast used |
| `status` | `ok`, or `insufficient_history` if the room has too little recent data |
| `outside_training_range` | `True` if an input lies beyond anything the tree was trained on |

The forecast is issued at the *end* of the `as_of` bin, so the lead time is the horizon minus one bin (5 minutes).

**Unusual input is handled explicitly.** Missing or non-numeric headcounts are dropped and counted (`result.attrs["dropped_readings"]`). Negative headcounts, missing columns, an empty table, and timestamps without a UTC offset raise `ValueError` (naive times are ambiguous around daylight saving). A room that has gone quiet after a near-empty reading is treated as empty, the same rule used in training; a room that went quiet while busy gets `insufficient_history` rather than a guess. A decision tree cannot extrapolate: outside its training range it repeats the value of its nearest leaf, so `outside_training_range` flags those forecasts as less trustworthy.

### The data used to build and test the model

Place these three files in `data/raw/` (see [`data/README.md`](data/README.md)). They are not stored in the repository (1.6 GB, supplied by the teaching team).

| File | Contents |
|---|---|
| `5occupancySensor_MayToDec2024_9MRows.csv` | 9,091,732 occupancy readings: room (`floorspaceid`), `headcount`, time with a UTC offset |
| `5EnvSensor_MayToDec2024_180kRows.csv` | Environmental readings (CO2, temperature and others) from 5 sensors; audited but not used for forecasting |
| `Sensor_ID_and_Locations.xlsx` | Sensor-to-room lookup; its occupancy sheet is labelled "Incorrect Data" by its authors |

## 3. Approach

```
raw readings -> 5-minute averages -> cleaned timeline -> features and target -> split by time -> fit and score -> forecast
 data_loading    data_loading          cleaning             features             validation      models, splines   forecast
```

1. **Load.** Average the raw readings into 5-minute bins in Melbourne local time (DuckDB streams the 1.6 GB file).
2. **Audit and clean.** Put each room on a complete timeline, cap spikes more than 5 people above a rolling median, interpolate gaps up to 15 minutes, and fill gaps up to one day that follow a near-empty room with zero (`assumed_empty`). Every step keeps the original value.
3. **Features.** For each room and time: headcount now and 15, 30 and 60 minutes ago, rolling means over 30 and 60 minutes, and the hour and day of week of the time being predicted. Targets are the headcount `h` minutes later. Rows whose target is missing are dropped.
4. **Split by time**, never at random: each room's history is split 60% train, 20% validation, 20% test, with a gap so that no training target falls inside the validation period.
5. **Methods compared**, all on identical rows: persistence; the headcount at the same time last week; a seasonal average by room, day and time of day; linear regression; a smoothing-spline daily profile; a spline hybrid (the daily profile plus a fitted fraction of today's deviation from it); and a **decision tree** (the main model).
6. **Choose and test.** Tree size is chosen on validation data, using training error against validation error. Every learned method is then refitted on training plus validation data, and the test set is scored once, under a protocol written beforehand ([`scripts/final_evaluation.py`](scripts/final_evaluation.py)). Uncertainty comes from a block bootstrap over whole room-days (2,000 resamples), because errors at neighbouring times are strongly correlated.

**Data flow at 30 minutes.** The 9,091,732 raw readings become 482,538 five-minute averages, which fill a timeline of 1,027,140 bins (100,271 left missing, because no defensible value exists for them). After dropping examples with a missing input or target, 867,719 examples remain; after the gap that keeps the splits apart, there are 520,600 training, 173,513 validation and 173,546 test examples. The tree chosen has depth 8 and 206 leaves (minimum leaf size 200).

**Sensitivity analysis.** Because there is no ground truth for why the sensor goes quiet, three choices were tested: the forecast horizon, the "quiet gaps are empty rooms" rule, and the bin width (1, 5 and 15 minutes). **Optimisation.** Tree tuning was the one slow step; it was made 8.5x faster with no change in the tree chosen, and the pipeline's runtime, memory and arithmetic cost were measured (see [`results/README.md`](results/README.md)).

## 4. Results

Scores are on the **test set**, 30-minute horizon unless stated, on rows whose target was actually measured. "Improvement" is the reduction in RMSE relative to persistence, with a 95% bootstrap interval. RMSE is in people.

| Horizon | Persistence RMSE | Decision tree | Spline hybrid | Linear regression |
|---|---|---|---|---|
| 15 minutes | 1.14 | 5.4% (3.7 to 7.3) | 5.9% (5.0 to 6.7) | 3.9% (0.9 to 7.2) |
| 30 minutes | 1.46 | 9.2% (8.0 to 10.6) | 9.4% (8.5 to 10.2) | 6.6% (4.2 to 9.8) |
| 60 minutes | 1.81 | 14.5% (12.2 to 16.3) | 13.6% (12.8 to 14.4) | 11.6% (10.2 to 13.1) |

- The **tree beats persistence at every horizon**, and the advantage grows with horizon. All intervals exclude zero.
- The **tree and the spline hybrid are statistically indistinguishable** on RMSE: their intervals overlap almost completely. The tree has the lower mean absolute error at all three horizons.
- The tree beats linear regression by 2.8% (0.5 to 4.5) at 30 minutes; at 15 minutes the interval includes zero (-0.9 to 3.5).
- Methods that rely on the weekly calendar ("same time last week", "seasonal average") score **worse than persistence at 15 and 30 minutes**: occupancy here does not repeat closely from week to week, and recent history is the most useful input.
- **Validation was optimistic:** at 30 minutes it gave an improvement of 11.7%, against 9.2% on the test set. That is expected, since settings were chosen on validation data and the later test period differs.
- **One room dominates the error.** Room G.20 is much harder to forecast (persistence RMSE 2.92 against 1.00 for the others on average), and the tree gains only 6.5% there against 9% to 16% elsewhere.
- **The conclusions do not depend on the choices that cannot be checked against ground truth.** Switching the "assumed empty" rule off removes more than half the training rows, yet moves the tree's RMSE on measured targets by only 0.011 people. About 29% of test rows contain a value that used look-ahead in cleaning; on strictly causal rows the tree's 30-minute improvement is 14.5%, so look-ahead is not what produces the gain.
- **In practical terms** the tree's typical miss is about 1.3 people at 30 minutes, in rooms averaging 1 to 2.5. The improvement is statistically clear but modest.
- **The live function** works on raw readings: with a day of history it produces a forecast for 98.7% of room-times, and matches the batch pipeline exactly for 97.6% of those.

The full evidence is in the notebooks ([data audit](notebooks/01_data_audit.ipynb), [daily patterns](notebooks/02_daily_patterns.ipynb), [model results](notebooks/03_model_results.ipynb)) and the tables in [`results/`](results/).

## 5. Known data issues

- The files are named "MayToDec2024", but the occupancy timestamps run from November 2023 to April 2026.
- One hub device (`deviceid`) reports for all five rooms; `floorspaceid` identifies the room.
- Timestamps carry a UTC offset. Hours of day must be taken in Australia/Melbourne local time; taking them from UTC shifts the daily pattern by 10 to 11 hours.
- Only 2 of the 5 occupancy rooms (G.20 and G.25) and 1 of the 5 environmental sensors appear in the location lookup, and no room is identified for both sensor types, so the two data sets cannot be joined by room.
- Only about half the 5-minute timeline has readings (36% to 58% per room). Gaps cluster overnight and usually follow a near-empty room, which suggests the hub stays quiet when nothing is happening. This is an inference from timing, not something the data prove.
- Some environmental variables (volatile organic compounds and some particulate-matter readings) are missing from a noticeable share of readings.

## 6. Limitations

- **One building.** Five rooms, one hub, one period; nothing here shows how the model would do elsewhere.
- **The sensor is the truth.** There is no independent headcount, so the project forecasts what the sensor reports.
- **Occupancy drifts.** Mean headcount fell over the period, so scores from a time-based split are specific to those months.
- **The filled-in zeros are an inference.** The results are insensitive to them for measured targets, but that does not show they are correct.
- **Trees do not extrapolate.** Beyond its training range the tree repeats its nearest leaf (flagged by `outside_training_range`).
- **Live use needs history.** With only three hours of readings the function can forecast for far fewer room-times than with a day; give it at least a day.
- **Practical gain is modest.** The tree and the spline hybrid are about equally accurate, and the extra complexity of the tree buys little.

## 7. Getting started

**Requirements.** Python 3.12 (tested on Linux with Python 3.12.3; 3.11 or newer is needed by the pinned numpy and pandas). Exact versions of every package are in [`requirements-lock.txt`](requirements-lock.txt); [`requirements.txt`](requirements.txt) lists the loose minimums, which have not been tested.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt
# put the three data files in data/raw/ (see data/README.md)
pytest                               # the test suite does not need the data
```

**Reproduce the results.** Each command writes its tables to `results/`; they take from seconds to a few minutes on one CPU. `train.py` also builds a cache of 5-minute averages in `data/processed/`, which the other scripts reuse.

```bash
python scripts/train.py --horizon 30                 # tuning curve, baselines, tree (validation set)
python scripts/sensitivity.py                        # horizon, gap rule and bin width
python scripts/spline_comparison.py                  # spline models and the hybrid
python scripts/profile_pipeline.py --parts stages learning splines cost
python scripts/profile_pipeline.py --parts tuning loader
python scripts/train_final.py --horizon 30           # fit the final model bundle (test set not used)
python scripts/check_live_vs_batch.py --n 300        # live function against the batch pipeline
python scripts/final_evaluation.py                   # the sealed test set: see the note below
python scripts/build_docs_and_reports.py             # rebuild docs/ and reports/
```

`final_evaluation.py` scores the test set under a protocol written before the test set was first opened. Re-running it reproduces the published table; changing any modelling choice after looking at it would make the test result less trustworthy.

**Forecast from raw readings.**

```python
import pandas as pd
from occupancy_forecast.forecast import forecast_headcount

readings = pd.read_csv("last_day.csv", usecols=["floorspaceid", "headcount", "collecteddate"])
print(forecast_headcount(readings, "models/final_h30.joblib"))
```

**Notebooks** need Jupyter (`pip install jupyter`, or open them in VS Code); start it from the repository root. They are saved with their outputs, so they also display on GitHub.

## 8. Repository layout

```
src/occupancy_forecast/   library: data_loading, audit, cleaning, features, validation, models, splines,
                          experiments, final, forecast, profiling
scripts/                  the commands above (train, sensitivity, spline_comparison, profile_pipeline,
                          train_final, check_live_vs_batch, final_evaluation, build_docs_and_reports)
tests/                    pytest tests for the library, scripts, notebooks, documentation and this README
notebooks/                01 data audit, 02 daily patterns, 03 model results (saved with outputs)
results/                  tables written by the scripts, described in results/README.md
docs/                     HTML documentation of the code, generated by pdoc (open docs/index.html)
reports/                  test report (HTML and XML) and coverage table
data/                     data instructions only; the raw data and caches are git-ignored
models/                   saved model bundles (git-ignored; created by scripts/train_final.py)
AI_USE_LOG.md             record of how AI was used
requirements.txt, requirements-lock.txt, pyproject.toml, LICENSE
```

## 9. Testing, documentation and reproducibility

- **Tests.** The suite has 285 tests with 98% line coverage of the library (see [`reports/test_report.html`](reports/test_report.html) and [`reports/coverage.txt`](reports/coverage.txt)). It covers the library; the scripts (each must start and print usage); the notebooks (valid, fully executed, no errors, no machine paths); the documentation (every public function documents its parameters, returns and exceptions, and, on the pinned toolchain, the HTML in `docs/` matches a fresh build); and this README (its numbers match `results/`, and its links and commands point at real files).
- **How the tests were checked.** Where a bug could quietly change a result (for example a feature that looks into the future, or a split that leaks), the code was deliberately broken to confirm a test fails.
- **Documentation.** [`docs/index.html`](docs/index.html) documents every module. To read it as a web page, open it in a browser after cloning, or enable GitHub Pages for the `/docs` folder.
- **Reproducibility.** Results, documentation and reports were produced in fresh virtual environments built from `requirements-lock.txt`; the documentation came out byte-identical in two independently built environments.

## 10. AI use

AI (Claude, by Anthropic) was used throughout this project, for explaining the brief, profiling the data, writing and reviewing code and tests, running analyses, and drafting documentation. Every material use is recorded in [`AI_USE_LOG.md`](AI_USE_LOG.md), with how it was checked, the errors it made, and the decisions that remained mine. The full reflection is in the written report.

## 11. Licence and data

The code is released under the MIT licence ([`LICENSE`](LICENSE)). The data belong to Monash University, were supplied by the teaching team for this unit, and are not covered by that licence or included in this repository. Do not redistribute them.
