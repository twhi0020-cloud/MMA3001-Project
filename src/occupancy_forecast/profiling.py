"""Timing, memory and cost-model helpers for the performance analysis.

Three kinds of tool, each used by ``scripts/profile_pipeline.py``:

* **Measurement** - wall-clock timing (:func:`time_call`), peak Python memory
  (:func:`peak_memory_mb`) and prediction latency (:func:`time_prediction_latency`).
  Timing and memory are measured in *separate* passes because ``tracemalloc`` slows
  allocation-heavy code and would distort the timings.
* **A cost model** - order-of-magnitude estimates of arithmetic operations and bytes moved
  (:func:`estimate_tree_fit_ops` and friends), plus a roofline comparison
  (:func:`roofline`) against a measured machine benchmark (:func:`machine_benchmark`).
  These are *estimates from stated assumptions*, to explain measured runtimes, not exact
  counts. A tree is built from comparisons, sorting and branching, so "FLOPs" here means
  generic arithmetic/comparison operations.
* **A reference implementation** - :func:`resample_occupancy_pandas` reproduces
  ``resample_occupancy`` with chunked pandas instead of DuckDB. It is the baseline for the
  loader comparison and an independent check that the DuckDB result is correct.

Note: ``tracemalloc`` sees memory allocated through Python and NumPy/pandas, but not memory
inside native engines such as DuckDB; use process-level peak memory for those.
"""

from __future__ import annotations

import math
import sys
import tracemalloc
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

LOCAL_TZ = "Australia/Melbourne"


# ----------------------------------------------------------------------------- measurement

def time_call(fn, *args, repeats: int = 1, **kwargs) -> tuple:
    """Call ``fn`` and report how long it took.

    Args:
        fn: Callable to time.
        *args: Positional arguments for ``fn``.
        repeats: Number of timed calls (at least 1). The result of the last call is returned.
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        ``(result, best_seconds, all_seconds)``; the best (smallest) time is the usual
        low-noise summary because slow runs reflect interference, not the code.

    Raises:
        ValueError: If ``repeats`` is not a positive integer.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError(f"repeats must be a positive integer, got {repeats!r}")
    times, result = [], None
    for _ in range(repeats):
        t0 = perf_counter()
        result = fn(*args, **kwargs)
        times.append(perf_counter() - t0)
    return result, min(times), times


def peak_memory_mb(fn, *args, **kwargs) -> tuple:
    """Call ``fn`` and report its peak Python-level memory use.

    Args:
        fn: Callable to run.
        *args: Positional arguments for ``fn``.
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        ``(result, peak_megabytes)``. Memory inside native engines (e.g. DuckDB) is not seen.
    """
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    base = tracemalloc.get_traced_memory()[0]
    try:
        result = fn(*args, **kwargs)
        peak = tracemalloc.get_traced_memory()[1] - base
    finally:
        if not was_tracing:
            tracemalloc.stop()
    return result, peak / 1e6


def peak_rss_mb() -> float | None:
    """Peak resident memory of the current process so far, in megabytes.

    Unlike :func:`peak_memory_mb` this includes memory used inside native engines such as
    DuckDB. It is a process-lifetime maximum, so measure one method per fresh process.

    On Linux the peak is read from ``VmHWM`` in ``/proc/self/status``. That counter starts
    fresh in every new process, whereas ``ru_maxrss`` is inherited from the parent across
    ``exec`` and would wrongly report the parent's earlier peak for a child process. Other
    platforms fall back to ``ru_maxrss``.

    Returns:
        Peak resident set size in MB, or ``None`` where it cannot be measured (e.g. Windows).
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return float(line.split()[1]) / 1e3  # kB -> MB
    except OSError:
        pass
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def time_prediction_latency(predict_fn, df: pd.DataFrame, batch_sizes=(1, 100, 10_000), repeats: int = 5) -> pd.DataFrame:
    """Measure how long a predictor takes for different numbers of rows per call.

    Args:
        predict_fn: Callable taking a DataFrame of rows and returning predictions.
        df: Rows to predict (the first ``batch_size`` rows of it are used).
        batch_sizes: Rows per call to try; each must not exceed ``len(df)``.
        repeats: Calls per batch size (the best time is reported).

    Returns:
        DataFrame with ``batch_size``, ``seconds_per_call`` and ``microseconds_per_row``.

    Raises:
        ValueError: If ``df`` is empty or a batch size is invalid or larger than ``df``.
    """
    if df.empty:
        raise ValueError("df must be non-empty")
    rows = []
    for b in batch_sizes:
        if isinstance(b, bool) or not isinstance(b, int) or b < 1 or b > len(df):
            raise ValueError(f"batch size must be an integer between 1 and {len(df)}, got {b!r}")
        batch = df.iloc[:b]
        _, best, _ = time_call(predict_fn, batch, repeats=repeats)
        rows.append({"batch_size": b, "seconds_per_call": best, "microseconds_per_row": best / b * 1e6})
    return pd.DataFrame(rows)


def systematic_subsample(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Keep every ``stride``-th row, so the sample stays spread over the whole time range.

    Args:
        df: Rows in time order.
        stride: Keep one row in every ``stride`` (1 keeps everything).

    Returns:
        The subsampled rows, order preserved.

    Raises:
        ValueError: If ``stride`` is not a positive integer.
    """
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError(f"stride must be a positive integer, got {stride!r}")
    return df.iloc[::stride]


# ----------------------------------------------------------------------------- cost model

def machine_benchmark(matrix_size: int = 1200, copy_megabytes: int = 160, repeats: int = 3) -> dict:
    """Measure what this machine can actually do, for comparison with the cost model.

    Args:
        matrix_size: Side of the square matrices multiplied to estimate peak GFLOP/s.
        copy_megabytes: Array size copied to estimate memory bandwidth.
        repeats: Repetitions of each benchmark (the best is reported).

    Returns:
        Dict with ``peak_gflops`` (double-precision matrix multiply), ``bandwidth_gb_s``
        (array copy, counting read plus write) and ``balance_flop_per_byte`` (their ratio:
        work with fewer operations per byte than this is limited by memory, not arithmetic).

    Raises:
        ValueError: If a size or repeat count is not positive.
    """
    for name, v in (("matrix_size", matrix_size), ("copy_megabytes", copy_megabytes), ("repeats", repeats)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{name} must be a positive integer, got {v!r}")
    a = np.random.default_rng(0).random((matrix_size, matrix_size))
    _, t_mm, _ = time_call(lambda: a @ a, repeats=repeats)
    peak_gflops = 2 * matrix_size**3 / t_mm / 1e9

    x = np.ones(copy_megabytes * 1_000_000 // 8)
    _, t_cp, _ = time_call(lambda: x.copy(), repeats=repeats)
    bandwidth = 2 * x.nbytes / t_cp / 1e9
    return {"peak_gflops": peak_gflops, "bandwidth_gb_s": bandwidth,
            "balance_flop_per_byte": peak_gflops / bandwidth}


def _positive(**values) -> None:
    for name, v in values.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v <= 0:
            raise ValueError(f"{name} must be a positive number, got {v!r}")


def estimate_tree_fit_ops(n_samples: int, n_features: int, depth: int) -> float:
    """Rough operation count to grow a regression tree of a given depth.

    Assumption: at every level each sample is visited once per feature; each visit costs
    about ``log2(n)`` comparisons for sorting plus about 6 arithmetic operations for the
    running sums and the variance-reduction criterion. The sort term is an upper bound
    (nodes get smaller with depth). Order-of-magnitude only.

    Args:
        n_samples: Training rows.
        n_features: Features considered at each split.
        depth: Tree depth.

    Returns:
        Estimated operations.

    Raises:
        ValueError: If an argument is not positive.
    """
    _positive(n_samples=n_samples, n_features=n_features, depth=depth)
    return float(n_samples * n_features * depth * (math.log2(n_samples) + 6))


def estimate_tree_fit_bytes(n_samples: int, n_features: int, depth: int) -> float:
    """Rough bytes moved while growing a tree.

    Assumption: per sample, feature and level, one 4-byte feature value, one 8-byte target
    and one 8-byte index are read (20 bytes). Order-of-magnitude only.

    Args:
        n_samples: Training rows.
        n_features: Features considered at each split.
        depth: Tree depth.

    Returns:
        Estimated bytes.

    Raises:
        ValueError: If an argument is not positive.
    """
    _positive(n_samples=n_samples, n_features=n_features, depth=depth)
    return float(n_samples * n_features * depth * 20)


def estimate_linear_fit_flops(n_samples: int, n_features: int) -> float:
    """Floating-point operations to fit ordinary least squares with an intercept.

    Forming the normal equations costs about ``2 n q^2`` and solving them ``(2/3) q^3``,
    with ``q = n_features + 1``.

    Args:
        n_samples: Training rows.
        n_features: Features (excluding the intercept).

    Returns:
        Estimated floating-point operations.

    Raises:
        ValueError: If an argument is not positive.
    """
    _positive(n_samples=n_samples, n_features=n_features)
    q = n_features + 1
    return float(2 * n_samples * q**2 + (2 / 3) * q**3)


def estimate_smoothing_spline_flops(n_knots: int, n_lambda_evaluations: int = 30, bandwidth: int = 4) -> float:
    """Rough floating-point operations to fit one GCV smoothing spline.

    Assumption: each trial smoothing strength needs a banded solve and the trace for GCV,
    about ``4 * n_knots * bandwidth^2`` operations, repeated for the number of strengths tried.

    Args:
        n_knots: Number of data points (knots) in the fit.
        n_lambda_evaluations: Smoothing strengths evaluated during GCV.
        bandwidth: Half-bandwidth of the banded system (cubic spline: 4).

    Returns:
        Estimated floating-point operations.

    Raises:
        ValueError: If an argument is not positive.
    """
    _positive(n_knots=n_knots, n_lambda_evaluations=n_lambda_evaluations, bandwidth=bandwidth)
    return float(n_lambda_evaluations * 4 * n_knots * bandwidth**2)


def arithmetic_intensity(operations: float, bytes_moved: float) -> float:
    """Operations performed per byte moved to or from memory.

    Args:
        operations: Arithmetic operations.
        bytes_moved: Bytes read plus written.

    Returns:
        ``operations / bytes_moved``.

    Raises:
        ValueError: If either argument is not positive.
    """
    _positive(operations=operations, bytes_moved=bytes_moved)
    return operations / bytes_moved


def roofline(operations: float, bytes_moved: float, peak_gflops: float, bandwidth_gb_s: float) -> dict:
    """Compare a workload with the machine's limits (a simple roofline model).

    Args:
        operations: Arithmetic operations.
        bytes_moved: Bytes moved.
        peak_gflops: Machine peak (GFLOP/s).
        bandwidth_gb_s: Machine memory bandwidth (GB/s).

    Returns:
        Dict with ``intensity`` (operations per byte), ``machine_balance`` (peak / bandwidth),
        ``regime`` (``"memory-bound"`` or ``"compute-bound"``), ``attainable_gflops`` (the
        roofline limit) and ``min_seconds`` (the fastest this workload could run).

    Raises:
        ValueError: If an argument is not positive.
    """
    _positive(peak_gflops=peak_gflops, bandwidth_gb_s=bandwidth_gb_s)
    ai = arithmetic_intensity(operations, bytes_moved)
    balance = peak_gflops / bandwidth_gb_s
    attainable = min(peak_gflops, ai * bandwidth_gb_s)
    return {
        "intensity": ai,
        "machine_balance": balance,
        "regime": "memory-bound" if ai < balance else "compute-bound",
        "attainable_gflops": attainable,
        "min_seconds": operations / (attainable * 1e9),
    }


# ----------------------------------------------------------------------------- reference loader

def resample_occupancy_pandas(
    csv_path: str | Path,
    bin_minutes: int = 5,
    tz: str = LOCAL_TZ,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Reference implementation of ``resample_occupancy`` using chunked pandas.

    Reads the CSV in chunks, converts timestamps to local time, bins them, and accumulates
    per-bin sums and counts across chunks (so memory use is bounded by the chunk size). It is
    slower than the DuckDB version, which is why that one is used; here it serves as the
    baseline in the loader comparison and as an independent correctness check.

    Args:
        csv_path: Path to the raw occupancy CSV.
        bin_minutes: Bin width in minutes (positive integer).
        tz: IANA timezone used to convert the UTC-offset timestamps to local time.
        chunksize: Rows read per chunk (positive integer).

    Returns:
        DataFrame with the same columns as ``resample_occupancy``: ``floorspaceid``,
        ``timestamp``, ``mean_headcount`` and ``n_obs``, sorted by room then time.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If ``bin_minutes`` or ``chunksize`` is not a positive integer.
    """
    for name, v in (("bin_minutes", bin_minutes), ("chunksize", chunksize)):
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise ValueError(f"{name} must be a positive integer, got {v!r}")
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Occupancy CSV not found: {path}")

    totals = None
    for chunk in pd.read_csv(path, usecols=["floorspaceid", "headcount", "collecteddate"], chunksize=chunksize):
        local = pd.to_datetime(chunk["collecteddate"], utc=True).dt.tz_convert(tz).dt.tz_localize(None)
        part = (
            chunk.assign(timestamp=local.dt.floor(f"{bin_minutes}min"))
            .groupby(["floorspaceid", "timestamp"])["headcount"]
            .agg(["sum", "count"])
        )
        totals = part if totals is None else totals.add(part, fill_value=0)

    if totals is None:
        return pd.DataFrame(columns=["floorspaceid", "timestamp", "mean_headcount", "n_obs"])
    out = totals.reset_index()
    out["mean_headcount"] = out["sum"] / out["count"]
    out["n_obs"] = out["count"].astype("int64")
    out["floorspaceid"] = out["floorspaceid"].astype(str)
    return out[["floorspaceid", "timestamp", "mean_headcount", "n_obs"]].sort_values(
        ["floorspaceid", "timestamp"]).reset_index(drop=True)
