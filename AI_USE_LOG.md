# AI use log

Running record for the report's AI reflection (brief section 7). Add an entry whenever AI materially contributes.

| Date | Tool | Task | How I checked it | Errors / unhelpful output | My decision |
|---|---|---|---|---|---|
| 2026-09-28 | Claude | Explained brief; profiled datasets; proposed scope | Re-ran queries myself; questioned the "occupancy prediction" framing | Initial framing was circular (predicting a value the sensor already reports) | Chose 30-min headcount forecasting; audit as a finding |
| 2026-09-28 | Claude | Repo skeleton + `data_loading.py` | Read code; ran pytest | Caught UTC vs local-hour bug in first exploratory query | Adopted explicit Melbourne timezone conversion |
| 2026-10-02 | Claude | Chose forecast horizon (15/30/60 min) | Reviewed hourly/daily pattern charts and lag-change magnitudes myself before deciding | None found | Set 30-min as primary horizon, 15/60-min as secondary comparisons |
| 2026-10-02 | Claude | Raised overfitting concern about the planned model | Pushed back on "the model just learns the data" framing until I understood train/test split properly | None found | Added a seasonal-average baseline specifically to test whether the model beats pure time-of-day periodicity |
| 2026-10-02 | Claude | Proposed primary model (initially gradient boosting) | Compared against my own Week 5.2 decision tree notebook | Gradient boosting wasn't actually taught in the unit; initial suggestion didn't match coursework | Switched primary model to single `DecisionTreeRegressor`, matching the taught validation workflow (max_depth/min_samples_leaf, train/validation RMSE) |
| 2026-10-02 | Claude | Explained decision tree train/predict mechanics | Checked explanation against notebook's own definitions (node, split, leaf) | None found | Confirmed understanding before building; will defend this unprompted in Q&A |
| 2026-10-02 | Claude | Proposed adding a spline comparison model | Confirmed splines were covered in numerical-methods lectures before agreeing | None found | Added spline (time-of-day pattern) as a second comparison model, bringing in the "numerical methods" half of the unit alongside ML |
