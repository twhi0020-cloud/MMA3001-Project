# Results tables

Small CSV tables written by the analysis scripts. All scores are on the **validation** set;
the test set has not been used. Regenerate with:

```bash
python scripts/train.py --horizon 30        # tuning_h30.csv, validation_comparison_h30.csv, permutation_importance_h30.csv
python scripts/sensitivity.py               # sensitivity_*.csv
python scripts/spline_comparison.py         # spline_*.csv
python scripts/profile_pipeline.py --parts stages learning splines cost   # performance tables
python scripts/profile_pipeline.py --parts tuning loader                   # slower: tuning_strategies, loader_comparison
```

| File | Contents |
|---|---|
| `tuning_h30.csv` | Tree tuning grid: train/validation RMSE, leaves, fit time, chosen row |
| `validation_comparison_h30.csv` | All methods on common validation rows, 30 min ahead |
| `permutation_importance_h30.csv` | Which features the chosen tree relies on |
| `sensitivity_horizon.csv` | 15 / 30 / 60 minute horizons |
| `sensitivity_gap_rule.csv` | Effect of the `assumed_empty` rule, scored on identical measured-target rows |
| `sensitivity_gap_rule_sizes.csv` | Rows available under each gap-rule setting |
| `sensitivity_bin_width.csv` | 1 / 5 / 15 minute bins: rows, runtime, accuracy (fixed tree: depth 8, leaf 200) |
| `spline_comparison.csv` | Linear / cubic / PCHIP / Akima / smoothing splines and the spline hybrid vs baselines and the tree, at 15 / 30 / 60 minutes |
| `spline_curve_diagnostics.csv` | How many fitted profiles would go below zero (before clipping), per spline type |
| `spline_hybrid_phi.csv` | Fraction of today's deviation from the normal profile still present at the target time, by horizon |
| `spline_variants.csv` | Spline slot width (15/30/60 min) and day grouping (day of week vs weekday/weekend) |
| `stage_timings.csv` | Wall-clock time and peak Python memory of each pipeline stage |
| `profile_top_functions.txt` | `cProfile` listing of the hottest functions (cleaning, features, one tree fit) |
| `tuning_strategies.csv` | Exhaustive grid vs subsampled / coarse-to-fine / fast tuning: fits, time, chosen tree, validation RMSE |
| `learning_curve.csv` | Validation error and fit time against the number of training rows |
| `loader_comparison.csv` | DuckDB vs chunked pandas for the raw CSV (outputs verified identical): time and process peak memory |
| `spline_padding.csv` | Cost and accuracy of shortening the spline's midnight wrap-around (`pad_hours`) |
| `cost_model.csv` | Estimated operations, bytes and arithmetic intensity against measured time, with roofline regime |
| `machine_benchmark.csv` | Measured peak GFLOP/s and memory bandwidth of the machine that produced these tables |
| `prediction_latency.csv` | Microseconds per row for the tree, linear regression and spline hybrid at several batch sizes |

Timings are specific to the machine that produced them (see `machine_benchmark.csv`); the ratios between methods are what carry over.
