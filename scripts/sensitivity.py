"""Sensitivity analysis for the headcount forecasting pipeline.

Three experiments, all scored on the **validation** set only (the test set stays sealed):

1. ``horizon``: forecast 15, 30 and 60 minutes ahead.
2. ``gaps``: how the ``assumed_empty`` rule affects forecasts of *observed* headcount
   (zero-fill off, and quiet thresholds of 0.5, 1.0 and 2.0 people). All conditions are
   scored on identical rows with measured targets.
3. ``bins``: 5- vs 15-minute (and optionally 1-minute) bins, with runtimes.

Usage (from the repository root)::

    python scripts/sensitivity.py                       # all three experiments
    python scripts/sensitivity.py --experiments gaps    # just one
    python scripts/sensitivity.py --bins 5 15 1         # include 1-minute bins

Absolute errors are not comparable between horizons or bin widths (the predicted quantity
changes), so those tables also report each method's improvement over persistence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occupancy_forecast.data_loading import resample_occupancy  # noqa: E402
from occupancy_forecast.experiments import (  # noqa: E402
    METHODS,
    compare_conditions,
    reference_boundaries,
    run_condition,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--occupancy-csv", type=Path,
                   default=ROOT / "data/raw/5occupancySensor_MayToDec2024_9MRows.csv")
    p.add_argument("--experiments", nargs="+", choices=["horizon", "gaps", "bins"],
                   default=["horizon", "gaps", "bins"])
    p.add_argument("--horizons", type=int, nargs="+", default=[15, 30, 60])
    p.add_argument("--bins", type=int, nargs="+", default=[5, 15])
    p.add_argument("--depths", type=int, nargs="+", default=[6, 8, 10, 12], help="tree max_depth grid")
    p.add_argument("--leaves", type=int, nargs="+", default=[50, 200], help="tree min_samples_leaf grid")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return p.parse_args(argv)


def load_binned(csv: Path, bin_minutes: int, cache_dir: Path) -> pd.DataFrame:
    cache = cache_dir / f"binned_{bin_minutes}min.parquet"
    if cache.is_file():
        return pd.read_parquet(cache)
    print(f"  resampling to {bin_minutes}-minute bins ...", flush=True)
    binned = resample_occupancy(csv, bin_minutes)
    cache_dir.mkdir(parents=True, exist_ok=True)
    binned.to_parquet(cache)
    return binned


def with_improvement(table: pd.DataFrame) -> pd.DataFrame:
    """Add each method's RMSE improvement over persistence (positive = better)."""
    out = table.copy()
    persistence = out.xs("persistence", level="method")["rmse"]
    conditions = out.index.get_level_values("condition")
    out["rmse_vs_persistence_%"] = (1 - out["rmse"] / persistence.reindex(conditions).to_numpy()) * 100
    return out


def show(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(df.round(3).to_string())


def experiment_horizon(binned, horizons, results_dir, grid):
    print("\n=== Experiment 1: forecast horizon ===", flush=True)
    bounds = reference_boundaries(binned)
    runs = []
    for h in horizons:
        t0 = perf_counter()
        r = run_condition(binned, f"h{h}", horizon_minutes=h, boundaries=bounds, **grid)
        print(f"  horizon {h} min: {r.n_train:,} train / {r.n_val:,} val rows, tree {r.best_params}, "
              f"{perf_counter() - t0:.0f} s", flush=True)
        runs.append(r)
    tables = []
    for r in runs:
        for observed_only in (True, False):
            t = with_improvement(compare_conditions([r], observed_only=observed_only))
            t["targets"] = "observed only" if observed_only else "all usable"
            tables.append(t)
    table = pd.concat(tables)
    show("Validation error by horizon (people)", table.reset_index().set_index(["targets", "condition", "method"]))
    table.to_csv(results_dir / "sensitivity_horizon.csv")


def experiment_gaps(binned, results_dir, grid):
    print("\n=== Experiment 2: the 'assumed empty' gap rule (30 min ahead) ===", flush=True)
    bounds = reference_boundaries(binned)
    settings = {
        "zero_fill_off": {"max_quiet_bins": 0},
        "threshold_0.5 (default)": {"quiet_threshold": 0.5},
        "threshold_1.0": {"quiet_threshold": 1.0},
        "threshold_2.0": {"quiet_threshold": 2.0},
    }
    runs = []
    for name, overrides in settings.items():
        t0 = perf_counter()
        r = run_condition(binned, name, clean_overrides=overrides, boundaries=bounds, **grid)
        print(f"  {name}: {r.n_train:,} train rows, tree {r.best_params}, {perf_counter() - t0:.0f} s", flush=True)
        runs.append(r)
    table = compare_conditions(runs, observed_only=True)
    table["rmse_vs_persistence_%"] = with_improvement(table)["rmse_vs_persistence_%"]
    show("Scored on identical rows with measured targets (people)", table)
    table.to_csv(results_dir / "sensitivity_gap_rule.csv")
    sizes = pd.DataFrame({r.name: {"train_rows": r.n_train, "val_rows": r.n_val} for r in runs}).T
    show("Rows available under each setting", sizes)
    sizes.to_csv(results_dir / "sensitivity_gap_rule_sizes.csv")


def experiment_bins(csv, bins, cache_dir, results_dir, grid):
    print("\n=== Experiment 3: bin width (30 min ahead) ===", flush=True)
    ref = load_binned(csv, 5, cache_dir)
    bounds = reference_boundaries(ref)
    rows = []
    for b in bins:
        t0 = perf_counter()
        binned = load_binned(csv, b, cache_dir)
        r = run_condition(binned, f"bin_{b}min", bin_minutes=b, boundaries=bounds, **grid)
        total = perf_counter() - t0
        t = with_improvement(compare_conditions([r], observed_only=True))
        row = {
            "bin_minutes": b, "grid_rows_after_binning": len(binned), "train_rows": r.n_train,
            "val_rows": r.n_val, "clean_s": r.seconds["clean"], "features_s": r.seconds["features"],
            "tuning_s": r.seconds["tuning"], "total_s": total,
            "persistence_rmse": t.loc[(r.name, "persistence"), "rmse"],
            "tree_rmse": t.loc[(r.name, "decision_tree"), "rmse"],
            "tree_vs_persistence_%": t.loc[(r.name, "decision_tree"), "rmse_vs_persistence_%"],
            "linear_vs_persistence_%": t.loc[(r.name, "linear_regression"), "rmse_vs_persistence_%"],
        }
        rows.append(row)
        print(f"  {b}-minute bins done in {total:.0f} s", flush=True)
    table = pd.DataFrame(rows).set_index("bin_minutes")
    out_path = results_dir / "sensitivity_bin_width.csv"
    if out_path.is_file():  # merge with earlier runs so widths can be run one at a time
        previous = pd.read_csv(out_path, index_col="bin_minutes")
        table = pd.concat([previous.drop(index=table.index, errors="ignore"), table]).sort_index()
    show("Bin width comparison (validation, observed targets)", table)
    table.to_csv(out_path)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    t_start = perf_counter()
    grid = {"depths": tuple(args.depths), "leaves": tuple(args.leaves)}
    binned5 = load_binned(args.occupancy_csv, 5, args.cache_dir)
    if "horizon" in args.experiments:
        experiment_horizon(binned5, args.horizons, args.results_dir, grid)
    if "gaps" in args.experiments:
        experiment_gaps(binned5, args.results_dir, grid)
    if "bins" in args.experiments:
        experiment_bins(args.occupancy_csv, args.bins, args.cache_dir, args.results_dir, grid)
    print(f"\nTotal time {perf_counter() - t_start:.0f} s. Tables written to {args.results_dir}/")


if __name__ == "__main__":
    main()
