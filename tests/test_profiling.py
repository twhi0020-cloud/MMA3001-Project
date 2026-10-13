"""Tests for occupancy_forecast.profiling."""

import math
import time

import numpy as np
import pandas as pd
import pytest

from occupancy_forecast.data_loading import resample_occupancy
from occupancy_forecast.profiling import (
    arithmetic_intensity,
    estimate_linear_fit_flops,
    estimate_smoothing_spline_flops,
    estimate_tree_fit_bytes,
    estimate_tree_fit_ops,
    machine_benchmark,
    peak_memory_mb,
    peak_rss_mb,
    resample_occupancy_pandas,
    roofline,
    systematic_subsample,
    time_call,
    time_prediction_latency,
)

HEADER = ('"id","deviceid","floorspaceid","occupancystatus","headcount",'
          '"collecteddate","occupancystatuschangedate","previousoccupancystatus"\n')


def _csv(path, rows):
    lines = [HEADER]
    for i, (room, hc, ts) in enumerate(rows):
        lines.append(f'{i},"dev","{room}","CurrentlyOccupied",{hc},"{ts}","{ts}","NotOccupied"\n')
    path.write_text("".join(lines))
    return path


# ---------- measurement ----------

def test_time_call_returns_result_and_sensible_times():
    result, best, times = time_call(lambda x: (time.sleep(0.02), x * 2)[1], 21, repeats=3)
    assert result == 42 and len(times) == 3
    assert best == min(times) and best >= 0.02


def test_time_call_rejects_bad_repeats():
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            time_call(lambda: None, repeats=bad)


def test_peak_memory_sees_a_large_allocation_and_not_a_small_one():
    _, big = peak_memory_mb(lambda: np.ones(10_000_000))            # 80 MB
    _, small = peak_memory_mb(lambda: np.ones(1_000))
    assert 70 < big < 120
    assert small < 1


def test_peak_rss_is_positive_and_grows_with_a_big_allocation():
    before = peak_rss_mb()
    if before is None:
        pytest.skip("resource module not available on this platform")
    block = np.ones(30_000_000)            # 240 MB
    after = peak_rss_mb()
    assert before > 10 and after >= before + 150
    del block


def test_peak_rss_of_a_fresh_child_is_not_inherited_from_a_big_parent():
    import subprocess
    import sys
    if peak_rss_mb() is None or not sys.platform.startswith("linux"):
        pytest.skip("needs /proc (Linux)")
    block = np.ones(60_000_000)                       # parent now holds ~480 MB
    parent_peak = peak_rss_mb()
    code = "from occupancy_forecast.profiling import peak_rss_mb; print(peak_rss_mb())"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                          env={**__import__("os").environ, "PYTHONPATH": "src"})
    child_peak = float(done.stdout.strip().splitlines()[-1])
    assert parent_peak > 400
    assert child_peak < parent_peak / 2               # the child did not inherit the parent's peak
    del block


def test_prediction_latency_table():
    df = pd.DataFrame({"x": np.arange(1000.0)})
    table = time_prediction_latency(lambda d: d["x"].to_numpy() * 2, df, batch_sizes=(1, 10, 1000), repeats=2)
    assert list(table["batch_size"]) == [1, 10, 1000]
    assert (table["seconds_per_call"] > 0).all()
    assert np.allclose(table["microseconds_per_row"], table["seconds_per_call"] / table["batch_size"] * 1e6)


def test_prediction_latency_validates_inputs():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError):
        time_prediction_latency(lambda d: d, df.iloc[0:0])
    for bad in (0, 3, 1.5):
        with pytest.raises(ValueError, match="batch size"):
            time_prediction_latency(lambda d: d, df, batch_sizes=(bad,))


def test_systematic_subsample_keeps_every_kth_row_in_order():
    df = pd.DataFrame({"i": range(10)})
    assert systematic_subsample(df, 3)["i"].tolist() == [0, 3, 6, 9]
    assert systematic_subsample(df, 1)["i"].tolist() == list(range(10))
    for bad in (0, -2, 1.5, True):
        with pytest.raises(ValueError):
            systematic_subsample(df, bad)


# ---------- cost model ----------

def test_tree_cost_estimates_match_the_stated_formula_and_scale_sensibly():
    ops = estimate_tree_fit_ops(1024, 8, 5)
    assert ops == pytest.approx(1024 * 8 * 5 * (math.log2(1024) + 6))
    assert estimate_tree_fit_ops(1024, 8, 10) == pytest.approx(2 * ops)           # linear in depth
    assert estimate_tree_fit_ops(2048, 8, 5) > 2 * ops                            # slightly super-linear in n
    assert estimate_tree_fit_bytes(1000, 8, 5) == 1000 * 8 * 5 * 20


def test_linear_and_spline_estimates_known_values():
    assert estimate_linear_fit_flops(100, 2) == pytest.approx(2 * 100 * 9 + (2 / 3) * 27)
    assert estimate_smoothing_spline_flops(288, 30, 4) == 30 * 4 * 288 * 16
    assert estimate_smoothing_spline_flops(576) == pytest.approx(2 * estimate_smoothing_spline_flops(288))


@pytest.mark.parametrize("fn,args", [
    (estimate_tree_fit_ops, (0, 8, 5)), (estimate_tree_fit_ops, (100, 8, -1)),
    (estimate_tree_fit_bytes, (100, 0, 5)), (estimate_linear_fit_flops, (0, 3)),
    (estimate_smoothing_spline_flops, (0,)), (arithmetic_intensity, (0, 10)), (arithmetic_intensity, (10, 0)),
])
def test_cost_functions_reject_non_positive_inputs(fn, args):
    with pytest.raises(ValueError):
        fn(*args)


def test_roofline_regimes_and_limits():
    mem = roofline(operations=1e9, bytes_moved=1e9, peak_gflops=50, bandwidth_gb_s=10)     # AI 1 < balance 5
    assert mem["regime"] == "memory-bound" and mem["attainable_gflops"] == pytest.approx(10)
    assert mem["min_seconds"] == pytest.approx(1e9 / 10e9)
    comp = roofline(operations=1e12, bytes_moved=1e9, peak_gflops=50, bandwidth_gb_s=10)   # AI 1000 > 5
    assert comp["regime"] == "compute-bound" and comp["attainable_gflops"] == pytest.approx(50)
    assert comp["machine_balance"] == pytest.approx(5)


def test_machine_benchmark_returns_plausible_numbers():
    m = machine_benchmark(matrix_size=300, copy_megabytes=20, repeats=2)
    assert m["peak_gflops"] > 0.1 and m["bandwidth_gb_s"] > 0.1
    assert m["balance_flop_per_byte"] == pytest.approx(m["peak_gflops"] / m["bandwidth_gb_s"])
    with pytest.raises(ValueError):
        machine_benchmark(matrix_size=0)


# ---------- reference loader ----------

ROWS = [
    ("A", 2, "2024-03-01 09:00:02+11"), ("A", 4, "2024-03-01 09:03:00+11"), ("A", 10, "2024-03-01 09:05:01+11"),
    ("B", 1, "2024-03-01 09:00:00+11"), ("B", 3, "2024-03-01 09:04:59+11"),
    ("A", 6, "2024-04-07 02:30:00+11"), ("A", 8, "2024-04-07 02:30:00+10"),   # daylight-saving overlap
    ("B", 5, "2024-07-02 18:11:42+10"),
]


def test_pandas_reference_matches_duckdb_loader(tmp_path):
    f = _csv(tmp_path / "occ.csv", ROWS)
    expected = resample_occupancy(f, bin_minutes=5)
    actual = resample_occupancy_pandas(f, bin_minutes=5, chunksize=3)        # forces several chunks
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_pandas_reference_is_independent_of_chunk_size_and_bin_width(tmp_path):
    f = _csv(tmp_path / "occ.csv", ROWS)
    for bins in (5, 15):
        small = resample_occupancy_pandas(f, bin_minutes=bins, chunksize=1)
        large = resample_occupancy_pandas(f, bin_minutes=bins, chunksize=1000)
        pd.testing.assert_frame_equal(small, large)
        pd.testing.assert_frame_equal(large, resample_occupancy(f, bin_minutes=bins), check_dtype=False)


def test_pandas_reference_bins_average_correctly(tmp_path):
    out = resample_occupancy_pandas(_csv(tmp_path / "occ.csv", ROWS[:3]), bin_minutes=5)
    assert out["mean_headcount"].tolist() == [3.0, 10.0] and out["n_obs"].tolist() == [2, 1]


def test_pandas_reference_validation(tmp_path):
    f = _csv(tmp_path / "occ.csv", ROWS[:1])
    with pytest.raises(FileNotFoundError):
        resample_occupancy_pandas(tmp_path / "nope.csv")
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            resample_occupancy_pandas(f, bin_minutes=bad)
        with pytest.raises(ValueError):
            resample_occupancy_pandas(f, chunksize=bad)
