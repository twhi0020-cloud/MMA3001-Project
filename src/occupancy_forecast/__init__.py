"""Forecast how many people will be in a room 15, 30 or 60 minutes ahead.

The package is a pipeline: each module does one stage, and the scripts in ``scripts/`` and the
notebooks in ``notebooks/`` join the stages together. Nothing here runs when it is imported;
the modules only define functions and classes.

| Stage | Module | What it does |
|---|---|---|
| Load | `occupancy_forecast.data_loading` | Averages millions of raw sensor readings into 5-minute bins in Melbourne local time |
| Audit | `occupancy_forecast.audit` | Reproducible data-quality checks (ID matching, date ranges, completeness) |
| Clean | `occupancy_forecast.cleaning` | Complete timeline, spike capping and gap filling; every step keeps the original value |
| Features | `occupancy_forecast.features` | One row per forecast: past headcounts, rolling means, calendar features and the target |
| Validate | `occupancy_forecast.validation` | Time-ordered train / validation / test split, and error metrics |
| Models | `occupancy_forecast.models`, `occupancy_forecast.splines` | Baselines, linear regression and the decision tree; spline profiles and the spline hybrid |
| Compare | `occupancy_forecast.experiments` | Run the pipeline under different settings and compare on identical rows |
| Finalise | `occupancy_forecast.final` | Final models, look-ahead flags and block-bootstrap uncertainty |
| Use | `occupancy_forecast.forecast` | `occupancy_forecast.forecast.forecast_headcount`: raw readings in, a forecast per room out |
| Performance | `occupancy_forecast.profiling` | Timing, memory, a cost model and a reference loader |

Data flow, from the raw file to a forecast::

    raw readings (CSV)  ->  5-minute bins  ->  cleaned timeline  ->  features and target
        data_loading            data_loading        cleaning              features
    ->  time-ordered split  ->  fit and score models  ->  forecast
            validation           models, splines          forecast

Example, with the final model saved by ``scripts/train_final.py``::

    from occupancy_forecast.forecast import forecast_headcount
    result = forecast_headcount(readings, "models/final_h30.joblib")

``readings`` is a table with ``floorspaceid``, ``headcount`` and ``collecteddate`` (a time with a
UTC offset); see `occupancy_forecast.forecast` for the exact input, output and error handling.
"""
