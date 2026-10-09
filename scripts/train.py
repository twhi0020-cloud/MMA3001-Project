"""Train and compare headcount forecasters end to end.

Pipeline: raw occupancy CSV -> 5-minute bins (Melbourne local time) -> cleaned grid ->
supervised table for one horizon -> chronological split -> baselines + tuned decision tree.

Only the **training** and **validation** sets are used here. The test set is held back
until the final model has been chosen, so that its score is an honest estimate.

Usage (from the repository root)::

    python scripts/train.py --horizon 30
    python scripts/train.py --horizon 60 --depths 4 8 12 --leaves 10 50

Outputs go to ``results/`` (tables) and ``models/`` (the fitted tree).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occupancy_forecast.cleaning import clean_occupancy, cleaning_summary  # noqa: E402
from occupancy_forecast.data_loading import resample_occupancy  # noqa: E402
from occupancy_forecast.features import build_supervised_dataset, feature_names  # noqa: E402
from occupancy_forecast.models import (  # noqa: E402
    SeasonalAverage,
    fit_decision_tree,
    fit_linear,
    permutation_importance_table,
    predict_last_week,
    predict_persistence,
    tune_decision_tree,
)
from occupancy_forecast.validation import chronological_split, compare_predictions  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--occupancy-csv", type=Path,
                   default=ROOT / "data/raw/5occupancySensor_MayToDec2024_9MRows.csv")
    p.add_argument("--horizon", type=int, default=30, help="forecast horizon in minutes")
    p.add_argument("--bin-minutes", type=int, default=5)
    p.add_argument("--depths", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12, 16, 20])
    p.add_argument("--leaves", type=int, nargs="+", default=[1, 10, 50, 200])
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed",
                   help="where the resampled table is cached (git-ignored)")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--models-dir", type=Path, default=ROOT / "models")
    return p.parse_args(argv)


def load_binned(csv: Path, bin_minutes: int, cache_dir: Path) -> pd.DataFrame:
    """Resample the raw CSV, reusing a cached copy if one exists."""
    cache = cache_dir / f"binned_{bin_minutes}min.parquet"
    if cache.is_file():
        print(f"Loading cached bins: {cache}")
        return pd.read_parquet(cache)
    print(f"Resampling {csv.name} (this takes ~20-30 s) ...")
    binned = resample_occupancy(csv, bin_minutes)
    cache_dir.mkdir(parents=True, exist_ok=True)
    binned.to_parquet(cache)
    return binned


def main(argv=None) -> None:
    args = parse_args(argv)
    t_start = perf_counter()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    binned = load_binned(args.occupancy_csv, args.bin_minutes, args.cache_dir)
    clean = clean_occupancy(binned, bin_minutes=args.bin_minutes)
    print("\nCleaning summary (share of grid bins per fill status):")
    print(cleaning_summary(clean).rename(index=lambda s: s[:8]).round(3).to_string())

    features = feature_names()
    data = build_supervised_dataset(clean, horizon_minutes=args.horizon, bin_minutes=args.bin_minutes)
    train, val, test = chronological_split(data)
    print(f"\nHorizon {args.horizon} min: {len(data):,} usable rows "
          f"(train {len(train):,} | validation {len(val):,} | test {len(test):,}, test held back)")

    # ---- baselines and linear model ----
    seasonal = SeasonalAverage().fit(train)
    linear = fit_linear(train, features)
    y_val = val["target"].to_numpy()
    predictions = {
        "persistence": predict_persistence(val),
        "same time last week": predict_last_week(val),
        "seasonal average": seasonal.predict(val),
        "linear regression": linear.predict(val[features]),
    }

    # ---- decision tree tuning ----
    print("\nTuning decision tree (train vs validation RMSE) ...")
    results, best = tune_decision_tree(train, val, features, args.depths, args.leaves)
    print(results.sort_values("val_rmse").head(8).round(4).to_string(index=False))
    print(f"\nChosen: {best}")
    tree = fit_decision_tree(train, features, **best)
    predictions["decision tree"] = tree.predict(val[features])

    # ---- comparisons (validation only) ----
    table = compare_predictions(y_val, predictions).sort_values("rmse")
    print(f"\nValidation results, rows where every method has a prediction (people):")
    print(table.round(3).to_string())

    observed = (val["target_fill_status"] == "observed").to_numpy()
    obs_table = compare_predictions(
        y_val[observed], {k: v[observed] for k, v in predictions.items()}
    ).sort_values("rmse")
    print("\nSame, but only rows whose target was actually observed (not assumed empty/interpolated):")
    print(obs_table.round(3).to_string())

    rooms = val["floorspaceid"].to_numpy()
    per_room = {}
    for room in sorted(set(rooms)):
        m = rooms == room
        t = compare_predictions(y_val[m], {k: v[m] for k, v in predictions.items()})
        per_room[room[:8]] = t["rmse"]
    print("\nValidation RMSE by room:")
    print(pd.DataFrame(per_room).T.round(3).to_string())

    importance = permutation_importance_table(tree, val, features)
    print("\nPermutation importance of the chosen tree (RMSE increase when shuffled, validation):")
    print(importance.round(4).to_string(index=False))

    # ---- save ----
    tag = f"h{args.horizon}"
    results.to_csv(args.results_dir / f"tuning_{tag}.csv", index=False)
    table.to_csv(args.results_dir / f"validation_comparison_{tag}.csv")
    importance.to_csv(args.results_dir / f"permutation_importance_{tag}.csv", index=False)
    joblib.dump({"model": tree, "features": features, "horizon_minutes": args.horizon, "params": best},
                args.models_dir / f"decision_tree_{tag}.joblib")
    print(f"\nSaved tables to {args.results_dir}/ and model to {args.models_dir}/. "
          f"Total time {perf_counter() - t_start:.0f} s.")


if __name__ == "__main__":
    main()
