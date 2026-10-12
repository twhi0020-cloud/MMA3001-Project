"""Compare time-of-day spline models with the baselines and the decision tree.

Everything is scored on the **validation** set only (the test set stays sealed), on identical
rows, so differences come from the model and not from the data.

Parts:

* ``horizons``: linear / cubic / PCHIP / Akima / smoothing splines and the spline hybrid
  against persistence, the seasonal average, linear regression and the tree, at 15, 30 and
  60 minutes. Also reports how often each spline construction would have gone negative.
* ``variants``: how the smoothing spline and hybrid change with slot width (15/30/60 min) and
  with day grouping (day of week vs weekday/weekend). The tree is not refitted (faster).

Usage (from the repository root)::

    python scripts/spline_comparison.py --parts horizons
    python scripts/spline_comparison.py --parts variants
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occupancy_forecast.experiments import (  # noqa: E402
    METHODS,
    SPLINE_COLUMNS,
    compare_conditions,
    reference_boundaries,
    run_condition,
)

ALL_METHODS = METHODS + SPLINE_COLUMNS


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parts", nargs="+", choices=["horizons", "variants"], default=["horizons", "variants"])
    p.add_argument("--horizons", type=int, nargs="+", default=[15, 30, 60])
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    return p.parse_args(argv)


def add_improvement(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    persistence = out.xs("persistence", level="method")["rmse"]
    cond = out.index.get_level_values("condition")
    out["rmse_vs_persistence_%"] = (1 - out["rmse"] / persistence.reindex(cond).to_numpy()) * 100
    return out


def part_horizons(binned, horizons, results_dir):
    print("\n=== Spline models vs baselines, by horizon ===", flush=True)
    bounds = reference_boundaries(binned)
    tables, diag, phis = [], [], {}
    for h in horizons:
        t0 = perf_counter()
        r = run_condition(binned, f"h{h}", horizon_minutes=h, boundaries=bounds, splines=True)
        print(f"  horizon {h} min done in {perf_counter() - t0:.0f} s "
              f"(splines fitted in {r.seconds['splines']:.1f} s)", flush=True)
        for observed_only in (True, False):
            t = add_improvement(compare_conditions([r], observed_only=observed_only, methods=ALL_METHODS))
            t["targets"] = "observed only" if observed_only else "all usable"
            tables.append(t)
        d = r.spline_diagnostics.assign(horizon=h)
        diag.append(d)
        phis[h] = r.spline_phi
    table = pd.concat(tables).reset_index().set_index(["targets", "condition", "method"])
    print("\nValidation error (people); rmse_vs_persistence_% > 0 means better than persistence")
    print(table.round(3).to_string())
    table.to_csv(results_dir / "spline_comparison.csv")

    d = pd.concat(diag)
    d = d[(d["horizon"] == horizons[0]) & d["day"].notna()]  # profiles do not depend on horizon enough to matter
    summary = d.groupby("method").agg(
        curves=("curve_min", "size"),
        lowest_curve_min=("curve_min", "min"),
        curves_going_negative=("curve_min", lambda x: int((x < 0).sum())),
    )
    print(f"\nCurve minima before clipping at zero ({int(summary['curves'].iloc[0])} room-day profiles per method)")
    print(summary.round(3).to_string())
    summary.to_csv(results_dir / "spline_curve_diagnostics.csv")

    phi = pd.Series(phis, name="phi").rename_axis("horizon_min")
    print("\nHybrid persistence fraction phi (share of today's deviation still present at the target time)")
    print(phi.round(3).to_string())
    phi.to_csv(results_dir / "spline_hybrid_phi.csv")


def part_variants(binned, results_dir):
    print("\n=== Spline settings (30 min ahead; tree not refitted) ===", flush=True)
    bounds = reference_boundaries(binned)
    variants = {
        "slot15_dow (default)": {"slot_minutes": 15, "day_grouping": "dow"},
        "slot30_dow": {"slot_minutes": 30, "day_grouping": "dow"},
        "slot60_dow": {"slot_minutes": 60, "day_grouping": "dow"},
        "slot15_weekday_weekend": {"slot_minutes": 15, "day_grouping": "weekday_weekend"},
        "slot30_weekday_weekend": {"slot_minutes": 30, "day_grouping": "weekday_weekend"},
    }
    runs = []
    for name, cfg in variants.items():
        t0 = perf_counter()
        runs.append(run_condition(binned, name, boundaries=bounds, splines=True,
                                  spline_config=cfg, fit_tree=False))
        print(f"  {name}: {perf_counter() - t0:.0f} s", flush=True)
    methods = ["persistence", "seasonal_average", "spline_pchip", "spline_smoothing", "spline_hybrid"]
    table = add_improvement(compare_conditions(runs, observed_only=True, methods=methods))
    wide = table["rmse"].unstack("method")[methods]
    print("\nRMSE (people) on identical rows with measured targets")
    print(wide.round(3).to_string())
    gain = table["rmse_vs_persistence_%"].unstack("method")[["spline_smoothing", "spline_hybrid"]]
    print("\nImprovement over persistence (%)")
    print(gain.round(2).to_string())
    table.to_csv(results_dir / "spline_variants.csv")


def main(argv=None) -> None:
    args = parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir / "binned_5min.parquet"
    if not cache.is_file():
        sys.exit(f"Missing {cache}; run scripts/train.py first to create it.")
    binned = pd.read_parquet(cache)
    t0 = perf_counter()
    if "horizons" in args.parts:
        part_horizons(binned, args.horizons, args.results_dir)
    if "variants" in args.parts:
        part_variants(binned, args.results_dir)
    print(f"\nTotal time {perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
