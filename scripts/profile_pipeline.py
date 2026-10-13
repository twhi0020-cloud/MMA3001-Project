"""Performance analysis of the forecasting pipeline: where the time goes and what helps.

Parts (each writes a table to ``results/``; all model scores are on the **validation** set):

* ``stages``   - wall-clock time and peak Python memory of each pipeline stage, plus a
  ``cProfile`` listing of the hottest functions.
* ``tuning``   - the exhaustive tree grid search (baseline) against faster strategies
  (subsampling, coarse-to-fine, and both), judged on time, number of fits and the tree chosen.
* ``learning`` - validation error and fit time against the amount of training data.
* ``loader``   - DuckDB (selected) against chunked pandas (baseline) for the raw CSV, with an
  equality check that the two give identical bins.
* ``splines``  - cost and accuracy of shortening the spline midnight wrap-around (``pad_hours``).
* ``cost``     - machine benchmark, FLOP / bytes / arithmetic-intensity estimates against
  measured times, and prediction latency.

Usage (from the repository root, with the cache from ``scripts/train.py`` present)::

    python scripts/profile_pipeline.py --parts stages learning cost
    python scripts/profile_pipeline.py --parts tuning --strategies exhaustive fast
    python scripts/profile_pipeline.py --parts loader

Timings depend on the machine; the tables record relative effects that should carry over.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import re
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from occupancy_forecast.cleaning import clean_occupancy  # noqa: E402
from occupancy_forecast.data_loading import resample_occupancy  # noqa: E402
from occupancy_forecast.features import build_supervised_dataset, feature_names  # noqa: E402
from occupancy_forecast.models import (  # noqa: E402
    SeasonalAverage,
    fit_decision_tree,
    fit_linear,
    tune_decision_tree,
    tune_decision_tree_fast,
)
from occupancy_forecast.profiling import (  # noqa: E402
    arithmetic_intensity,
    estimate_linear_fit_flops,
    estimate_smoothing_spline_flops,
    estimate_tree_fit_bytes,
    estimate_tree_fit_ops,
    machine_benchmark,
    peak_memory_mb,
    resample_occupancy_pandas,
    roofline,
    systematic_subsample,
    time_call,
    time_prediction_latency,
)
from occupancy_forecast.splines import SplineAnomalyForecaster, SplineProfile  # noqa: E402
from occupancy_forecast.validation import chronological_split  # noqa: E402

STRATEGIES = ["exhaustive", "subsample25", "subsample10", "coarse_to_fine", "fast"]
GRID_DEPTHS = [2, 4, 6, 8, 10, 12, 16, 20]
GRID_LEAVES = [1, 10, 50, 200]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parts", nargs="+", choices=["stages", "tuning", "learning", "loader", "splines", "cost"],
                   default=["stages", "learning", "cost"])
    p.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=STRATEGIES)
    p.add_argument("--occupancy-csv", type=Path, default=ROOT / "data/raw/5occupancySensor_MayToDec2024_9MRows.csv")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--sample-rows", type=int, default=1_000_000, help="rows used for the loader comparison")
    return p.parse_args(argv)


def load_split(cache_dir: Path, horizon: int = 30):
    cache = cache_dir / "binned_5min.parquet"
    if not cache.is_file():
        sys.exit(f"Missing {cache}; run scripts/train.py first to create it.")
    binned = pd.read_parquet(cache)
    clean = clean_occupancy(binned)
    train, val, _ = chronological_split(build_supervised_dataset(clean, horizon))
    return binned, clean, train, val


def rmse_of(model, val, features) -> float:
    return float(np.sqrt(np.mean((model.predict(val[features]) - val["target"].to_numpy()) ** 2)))


def merge_csv(path: Path, new: pd.DataFrame, key: str) -> pd.DataFrame:
    """Merge ``new`` into an existing table by ``key`` so parts can be run separately."""
    if path.is_file():
        old = pd.read_csv(path, index_col=key)
        new = pd.concat([old.drop(index=new.index, errors="ignore"), new])
    new.to_csv(path)
    return new


# ------------------------------------------------------------------------------ stages

def part_stages(args):
    print("\n=== Stage timings and memory (30-minute horizon) ===", flush=True)
    csv = args.occupancy_csv
    features = feature_names()
    rows = []

    def record(name, fn, rows_out=None):
        _, seconds, _ = time_call(fn, repeats=1)
        _, peak = peak_memory_mb(fn) if seconds < 15 else (None, float("nan"))
        rows.append({"stage": name, "seconds": seconds, "peak_python_mb": peak})
        print(f"  {name:34s} {seconds:7.2f} s   {peak:8.0f} MB", flush=True)

    if csv.is_file():
        resample_occupancy(csv, 5)  # warm-up read, so the timed run is not a cold-disk read
        record("1 resample raw CSV (DuckDB)", lambda: resample_occupancy(csv, 5))
    binned = pd.read_parquet(args.cache_dir / "binned_5min.parquet")
    state = {}
    record("2 clean (grid, spikes, gaps)", lambda: state.update(clean=clean_occupancy(binned)))
    record("3 build features and targets", lambda: state.update(data=build_supervised_dataset(state["clean"], 30)))
    record("4 chronological split", lambda: state.update(split=chronological_split(state["data"])))
    train, val, _ = state["split"]
    record("5 seasonal average (fit)", lambda: SeasonalAverage().fit(train))
    record("6 linear regression (fit)", lambda: fit_linear(train, features))
    record("7 decision tree (1 fit, d8 leaf200)", lambda: fit_decision_tree(train, features, 8, 200))
    holder = {}
    record("8 spline smoothing profile (fit)", lambda: holder.update(profile=SplineProfile("smoothing").fit(train)))
    record("9 spline hybrid (fit, own profile)", lambda: SplineAnomalyForecaster(method="smoothing").fit(train))
    record("10 spline hybrid (fit, reusing profile)",
           lambda: SplineAnomalyForecaster(method="smoothing").fit(train, profile=holder["profile"]))
    table = pd.DataFrame(rows).set_index("stage")
    table.to_csv(args.results_dir / "stage_timings.csv")

    print("\n  cProfile of stages 2-4 and 7 (top functions by cumulative time):")
    pr = cProfile.Profile()
    pr.enable()
    c = clean_occupancy(binned)
    tr, _, _ = chronological_split(build_supervised_dataset(c, 30))
    fit_decision_tree(tr, features, 8, 200)
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(18)
    text = buf.getvalue()
    (args.results_dir / "profile_top_functions.txt").write_text(text)
    print("\n".join(text.splitlines()[:30]))


# ------------------------------------------------------------------------------ tuning

def part_tuning(args):
    print("\n=== Tuning strategies (30-minute horizon, validation set) ===", flush=True)
    _, _, train, val = load_split(args.cache_dir)
    features = feature_names()
    out_path = args.results_dir / "tuning_strategies.csv"

    def run(name):
        t0 = perf_counter()
        if name == "exhaustive":
            res, best = tune_decision_tree(train, val, features, GRID_DEPTHS, GRID_LEAVES)
        elif name in ("subsample25", "subsample10"):
            stride = 4 if name == "subsample25" else 10
            res, best = tune_decision_tree(systematic_subsample(train, stride), val, features, GRID_DEPTHS, GRID_LEAVES)
        elif name == "coarse_to_fine":
            res, best = tune_decision_tree_fast(train, val, features, stride=1)
        else:
            res, best = tune_decision_tree_fast(train, val, features, stride=4)
        tuned = perf_counter() - t0
        t1 = perf_counter()
        model = fit_decision_tree(train, features, **best)
        refit = perf_counter() - t1
        return {
            "strategy": name, "fits": len(res), "tune_seconds": tuned, "refit_seconds": refit,
            "total_seconds": tuned + refit, "max_depth": best["max_depth"],
            "min_samples_leaf": best["min_samples_leaf"], "n_leaves": int(model.get_n_leaves()),
            "val_rmse": rmse_of(model, val, features),
        }

    rows = []
    for name in args.strategies:
        r = run(name)
        rows.append(r)
        print(f"  {name:15s} {r['fits']:>2} fits  {r['total_seconds']:6.1f} s  -> depth {r['max_depth']}, "
              f"leaf {r['min_samples_leaf']}, val RMSE {r['val_rmse']:.4f}", flush=True)
    table = merge_csv(out_path, pd.DataFrame(rows).set_index("strategy"), "strategy")
    if "exhaustive" in table.index:
        table["speedup_vs_exhaustive"] = table.loc["exhaustive", "total_seconds"] / table["total_seconds"]
        table["val_rmse_vs_exhaustive_%"] = (table["val_rmse"] / table.loc["exhaustive", "val_rmse"] - 1) * 100
        table.to_csv(out_path)
    print("\n", table.round(4).to_string())


def part_learning(args):
    print("\n=== Learning curve: accuracy and cost against training size (depth 8, leaf 200) ===", flush=True)
    _, _, train, val = load_split(args.cache_dir)
    features = feature_names()
    rows = []
    for stride in (50, 20, 10, 4, 2, 1):
        sub = systematic_subsample(train, stride)
        model, seconds, _ = time_call(fit_decision_tree, sub, features, 8, 200, repeats=2)
        rows.append({"train_rows": len(sub), "fraction": len(sub) / len(train), "fit_seconds": seconds,
                     "train_rmse": rmse_of(model, sub, features), "val_rmse": rmse_of(model, val, features)})
    table = pd.DataFrame(rows).set_index("train_rows")
    table["val_rmse_vs_full_%"] = (table["val_rmse"] / table["val_rmse"].iloc[-1] - 1) * 100
    table.to_csv(args.results_dir / "learning_curve.csv")
    print(table.round(4).to_string())


# ------------------------------------------------------------------------------ loader

_CHILD = """
import json, sys, time
sys.path.insert(0, {src!r})
from occupancy_forecast.profiling import peak_rss_mb, resample_occupancy_pandas
from occupancy_forecast.data_loading import resample_occupancy
fn = {{"duckdb": resample_occupancy, "pandas": resample_occupancy_pandas}}[{method!r}]
t = time.perf_counter(); out = fn({csv!r}, 5); dt = time.perf_counter() - t
print(json.dumps({{"seconds": dt, "rows_out": len(out), "peak_rss_mb": peak_rss_mb()}}))
"""


def _run_loader(method: str, csv: Path) -> dict:
    code = _CHILD.format(src=str(SRC), method=method, csv=str(csv))
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    # DuckDB may print a progress bar on long runs, so take the last {...} in the output
    return json.loads(re.findall(r"\{.*\}", done.stdout)[-1])


def part_loader(args):
    print("\n=== Loader: DuckDB (selected) vs chunked pandas (baseline) ===", flush=True)
    csv = args.occupancy_csv
    if not csv.is_file():
        sys.exit(f"Missing {csv}")
    n = args.sample_rows
    total_rows = sum(1 for _ in open(csv, "rb")) - 1
    sample = args.cache_dir / f"sample_{n}.csv"
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with open(csv, "rb") as src, open(sample, "wb") as dst:
        for i, line in enumerate(src):
            dst.write(line)
            if i >= n:
                break

    # correctness: both implementations must give identical bins on the sample
    a, b = resample_occupancy(sample, 5), resample_occupancy_pandas(sample, 5)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)
    print(f"  equality check on {n:,} rows: DuckDB and pandas outputs are identical ({len(a):,} bins)", flush=True)

    rows = []
    for method in ("duckdb", "pandas"):
        r = _run_loader(method, sample)
        rows.append({"method": method, "rows": n, "seconds": r["seconds"], "peak_rss_mb": r["peak_rss_mb"],
                     "full_file_seconds_estimate": r["seconds"] * total_rows / n,
                     "full_file_peak_rss_mb_estimate": None})
        print(f"  {method:7s} {n:,} rows: {r['seconds']:6.1f} s, peak {r['peak_rss_mb']:.0f} MB", flush=True)
    full = _run_loader("duckdb", csv)
    rows.append({"method": "duckdb (measured, full file)", "rows": total_rows, "seconds": full["seconds"],
                 "peak_rss_mb": full["peak_rss_mb"], "full_file_seconds_estimate": full["seconds"],
                 "full_file_peak_rss_mb_estimate": full["peak_rss_mb"]})
    print(f"  duckdb  {total_rows:,} rows (full file): {full['seconds']:.1f} s, peak {full['peak_rss_mb']:.0f} MB")
    table = pd.DataFrame(rows).set_index("method")
    table["speedup_vs_pandas"] = table.loc["pandas", "seconds"] / table["seconds"]
    table.loc["duckdb (measured, full file)", "speedup_vs_pandas"] = (
        table.loc["pandas", "full_file_seconds_estimate"] / full["seconds"])
    table.to_csv(args.results_dir / "loader_comparison.csv")
    sample.unlink()
    print("\n", table.round(2).to_string())


# ------------------------------------------------------------------------------ splines

def part_splines(args):
    print("\n=== Spline midnight wrap-around: cost and accuracy of a shorter wrap (30 min ahead) ===", flush=True)
    _, _, train, val = load_split(args.cache_dir)
    y = val["target"].to_numpy()
    reference, rows = None, []
    for pad in (24.0, 12.0, 6.0, 3.0, 1.5):
        profile, seconds, _ = time_call(lambda: SplineProfile("smoothing", pad_hours=pad).fit(train), repeats=2)
        pred = profile.predict(val)
        hybrid = SplineAnomalyForecaster(method="smoothing", pad_hours=pad).fit(train, profile=profile)
        hpred = hybrid.predict(val)
        if reference is None:
            reference = (pred, hpred)
        rows.append({
            "pad_hours": pad, "fit_seconds": seconds,
            "max_abs_diff_vs_24h": float(np.abs(pred - reference[0]).max()),
            "profile_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "hybrid_rmse": float(np.sqrt(np.mean((hpred - y) ** 2))),
            "hybrid_max_abs_diff_vs_24h": float(np.abs(hpred - reference[1]).max()),
        })
    table = pd.DataFrame(rows).set_index("pad_hours")
    table["speedup_vs_24h"] = table.loc[24.0, "fit_seconds"] / table["fit_seconds"]
    table.to_csv(args.results_dir / "spline_padding.csv")
    print(table.round(5).to_string())


# ------------------------------------------------------------------------------ cost

def part_cost(args):
    print("\n=== Cost model against measured times ===", flush=True)
    _, _, train, val = load_split(args.cache_dir)
    features = feature_names()
    n, p = len(train), len(features)
    m = machine_benchmark()
    pd.Series(m, name="value").rename_axis("metric").to_csv(args.results_dir / "machine_benchmark.csv")
    print(f"  machine: {m['peak_gflops']:.1f} GFLOP/s peak, {m['bandwidth_gb_s']:.1f} GB/s, "
          f"balance {m['balance_flop_per_byte']:.1f} FLOP/byte")

    rows = []

    def add(name, ops, nbytes, seconds, note):
        r = roofline(ops, nbytes, m["peak_gflops"], m["bandwidth_gb_s"]) if nbytes else None
        rows.append({
            "operation": name, "est_operations": ops, "est_bytes": nbytes or np.nan, "measured_seconds": seconds,
            "achieved_gops": ops / seconds / 1e9, "percent_of_peak": ops / seconds / 1e9 / m["peak_gflops"] * 100,
            "arithmetic_intensity": arithmetic_intensity(ops, nbytes) if nbytes else np.nan,
            "regime": r["regime"] if r else "n/a", "roofline_min_seconds": r["min_seconds"] if r else np.nan,
            "note": note})

    _, t_tree, _ = time_call(fit_decision_tree, train, features, 8, 200, repeats=2)
    add("decision tree fit (d8)", estimate_tree_fit_ops(n, p, 8), estimate_tree_fit_bytes(n, p, 8), t_tree,
        f"n={n:,}, p={p}, depth 8; comparisons/branching, not pure FLOPs")
    _, t_deep, _ = time_call(fit_decision_tree, train, features, 20, 1, repeats=1)
    add("decision tree fit (d20, leaf 1)", estimate_tree_fit_ops(n, p, 20), estimate_tree_fit_bytes(n, p, 20), t_deep,
        "deepest tuning-grid tree; estimate treats every level as full size")
    _, t_lin, _ = time_call(fit_linear, train, features, repeats=3)
    add("linear regression fit", estimate_linear_fit_flops(n, p), None, t_lin, "normal equations; trivial cost")
    sp = SplineProfile("smoothing")
    _, t_sp, _ = time_call(sp.fit, train, repeats=2)
    n_curves = len(sp._curves)
    add("smoothing splines, all room/day profiles", n_curves * estimate_smoothing_spline_flops(3 * 96), None, t_sp,
        f"{n_curves} curves of {3 * 96} padded knots; time includes slot averaging")
    table = pd.DataFrame(rows).set_index("operation")
    table.to_csv(args.results_dir / "cost_model.csv")
    pd.set_option("display.width", 220)
    print(table.drop(columns="note").round(3).to_string())

    print("\n  Prediction latency (microseconds per row):")
    features_df = val
    tree = fit_decision_tree(train, features, 8, 200)
    linear = fit_linear(train, features)
    hybrid = SplineAnomalyForecaster(method="smoothing").fit(train)
    parts = []
    for name, fn in (("decision tree", lambda d: tree.predict(d[features])),
                     ("linear regression", lambda d: linear.predict(d[features])),
                     ("spline hybrid", hybrid.predict)):
        t = time_prediction_latency(fn, features_df, batch_sizes=(1, 100, 10_000, 100_000), repeats=5)
        parts.append(t.assign(model=name))
    lat = pd.concat(parts).set_index(["model", "batch_size"])
    lat.to_csv(args.results_dir / "prediction_latency.csv")
    print(lat.round(2).to_string())


def main(argv=None):
    args = parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    for part in args.parts:
        {"stages": part_stages, "tuning": part_tuning, "learning": part_learning,
         "loader": part_loader, "splines": part_splines, "cost": part_cost}[part](args)
    print(f"\nTotal time {perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
