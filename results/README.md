# Results tables

Small CSV tables written by the analysis scripts. All scores are on the **validation** set;
the test set has not been used. Regenerate with:

```bash
python scripts/train.py --horizon 30        # tuning_h30.csv, validation_comparison_h30.csv, permutation_importance_h30.csv
python scripts/sensitivity.py               # sensitivity_*.csv
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
