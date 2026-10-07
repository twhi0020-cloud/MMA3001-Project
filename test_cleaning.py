"""Tests for occupancy_forecast.cleaning."""

import numpy as np
import pandas as pd
import pytest

from occupancy_forecast.cleaning import (
    FILL_ASSUMED_EMPTY,
    FILL_INTERPOLATED,
    FILL_MISSING,
    FILL_OBSERVED,
    cap_spikes,
    clean_occupancy,
    cleaning_summary,
    fill_gaps,
    regularize_grid,
)

T0 = pd.Timestamp("2024-03-04 09:00:00")


def _binned(room, values, start=T0, step_min=5):
    """Build a resample_occupancy-style frame; NaN values are omitted (no readings)."""
    rows = [
        (room, start + pd.Timedelta(minutes=step_min * i), v, 3)
        for i, v in enumerate(values)
        if not np.isnan(v)
    ]
    return pd.DataFrame(rows, columns=["floorspaceid", "timestamp", "mean_headcount", "n_obs"])


def _grid(values, room="A"):
    """A regularised frame with the given values (NaN allowed)."""
    return regularize_grid(_binned(room, values))


# ---------- regularize_grid ----------

def test_regularize_adds_missing_bins_with_nan_and_zero_obs():
    out = _grid([1.0, np.nan, np.nan, 4.0])
    assert len(out) == 4
    assert out["mean_headcount_raw"].isna().tolist() == [False, True, True, False]
    assert out["n_obs"].tolist() == [3, 0, 0, 3]


def test_regularize_keeps_rooms_independent():
    df = pd.concat([
        _binned("A", [1.0, 1.0, 1.0]),
        _binned("B", [2.0, np.nan, 2.0], start=T0 + pd.Timedelta(hours=5)),
    ])
    out = regularize_grid(df)
    assert out.groupby("floorspaceid").size().to_dict() == {"A": 3, "B": 3}
    # B's grid must not extend back to A's start
    assert out[out.floorspaceid == "B"]["timestamp"].min() == T0 + pd.Timedelta(hours=5)


def test_regularize_rejects_misaligned_timestamps():
    df = _binned("A", [1.0, 1.0])
    df.loc[1, "timestamp"] += pd.Timedelta(minutes=2)
    with pytest.raises(ValueError, match="not aligned"):
        regularize_grid(df)


def test_regularize_rejects_duplicates():
    df = _binned("A", [1.0, 1.0])
    df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
    with pytest.raises(ValueError, match="duplicate"):
        regularize_grid(df)


def test_regularize_missing_columns_and_bad_bin():
    with pytest.raises(ValueError, match="missing required columns"):
        regularize_grid(pd.DataFrame({"floorspaceid": ["A"]}))
    with pytest.raises(ValueError):
        regularize_grid(_binned("A", [1.0]), bin_minutes=0)


def test_regularize_empty_input_returns_empty_frame():
    out = regularize_grid(_binned("A", [np.nan]))
    assert out.empty and "mean_headcount_raw" in out.columns


# ---------- cap_spikes ----------

def test_cap_spikes_flags_and_caps_single_outlier_only():
    values = [2.0] * 60
    values[30] = 50.0
    out = cap_spikes(_grid(values))
    assert out["is_spike"].sum() == 1
    assert bool(out.loc[30, "is_spike"])
    assert out.loc[30, "headcount_capped"] < 50.0
    assert out.loc[30, "headcount_capped"] == pytest.approx(2.0 + 5.0 * 1.0)  # median + n_sigma*min_scale
    assert out.loc[30, "mean_headcount_raw"] == 50.0  # raw is preserved
    untouched = out.drop(index=30)
    assert (untouched["headcount_capped"] == untouched["mean_headcount_raw"]).all()


def test_cap_spikes_does_not_flag_ordinary_fluctuation():
    rng = np.random.default_rng(0)
    values = list(2.0 + rng.normal(0, 0.5, 80))
    out = cap_spikes(_grid(values))
    assert out["is_spike"].sum() == 0


def test_cap_spikes_never_flags_missing_bins():
    values = [2.0] * 30 + [np.nan] * 5 + [2.0] * 30
    out = cap_spikes(_grid(values))
    assert not out["is_spike"].any()


def test_cap_spikes_requires_enough_neighbours():
    # Only 3 observed bins in the window -> below min_periods, nothing is flagged
    out = cap_spikes(_grid([1.0, 100.0, 1.0]))
    assert not out["is_spike"].any()


def test_cap_spikes_larger_n_sigma_flags_fewer():
    values = [2.0] * 60
    values[30] = 9.0  # 7 above median
    loose = cap_spikes(_grid(values), n_sigma=5.0)["is_spike"].sum()
    strict = cap_spikes(_grid(values), n_sigma=10.0)["is_spike"].sum()
    assert loose == 1 and strict == 0


def test_cap_spikes_rejects_irregular_grid_and_bad_params():
    irregular = _binned("A", [1.0] * 20)
    irregular = irregular.drop(index=5).rename(columns={"mean_headcount": "mean_headcount_raw"})
    with pytest.raises(ValueError, match="regular"):
        cap_spikes(irregular)
    with pytest.raises(ValueError):
        cap_spikes(_grid([1.0] * 20), window_bins=0)
    with pytest.raises(ValueError):
        cap_spikes(_grid([1.0] * 20), n_sigma=-1)
    with pytest.raises(ValueError, match="regularize_grid"):
        cap_spikes(pd.DataFrame({"floorspaceid": ["A"], "timestamp": [T0]}))


# ---------- fill_gaps ----------

def _filled(values, **kw):
    g = _grid(values)
    g["headcount_capped"] = g["mean_headcount_raw"]
    return fill_gaps(g, **kw)


def test_short_gap_is_linearly_interpolated():
    out = _filled([1.0, np.nan, np.nan, 4.0])
    assert out["headcount"].tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])
    assert out["fill_status"].tolist() == [FILL_OBSERVED, FILL_INTERPOLATED, FILL_INTERPOLATED, FILL_OBSERVED]


def test_long_gap_after_low_headcount_is_assumed_empty():
    out = _filled([0.2] + [np.nan] * 10 + [3.0])
    assert (out.loc[1:10, "headcount"] == 0.0).all()
    assert (out.loc[1:10, "fill_status"] == FILL_ASSUMED_EMPTY).all()


def test_long_gap_after_high_headcount_stays_missing():
    out = _filled([4.0] + [np.nan] * 10 + [3.0])
    assert out.loc[1:10, "headcount"].isna().all()
    assert (out.loc[1:10, "fill_status"] == FILL_MISSING).all()


def test_gap_longer_than_max_quiet_stays_missing():
    out = _filled([0.0] + [np.nan] * 20 + [0.0], max_quiet_bins=10)
    assert (out.loc[1:20, "fill_status"] == FILL_MISSING).all()


def test_gap_at_exact_interp_limit_is_interpolated():
    out = _filled([2.0, np.nan, np.nan, np.nan, 2.0], max_interp_bins=3)
    assert (out.loc[1:3, "fill_status"] == FILL_INTERPOLATED).all()


def test_observed_values_are_never_changed():
    values = [1.0, 2.0, np.nan, 5.0, 0.0, np.nan, np.nan, np.nan, np.nan, 1.0]
    out = _filled(values)
    observed = out["fill_status"] == FILL_OBSERVED
    assert out.loc[observed, "headcount"].tolist() == out.loc[observed, "headcount_capped"].tolist()


def test_fill_rooms_do_not_leak_into_each_other():
    a = _binned("A", [5.0, 5.0, 5.0])
    b = _binned("B", [0.0, np.nan, np.nan, np.nan, np.nan, 1.0])
    g = regularize_grid(pd.concat([a, b]))
    g["headcount_capped"] = g["mean_headcount_raw"]
    out = fill_gaps(g)
    assert (out[out.floorspaceid == "A"]["fill_status"] == FILL_OBSERVED).all()


def test_fill_gaps_validates_inputs():
    g = _grid([1.0, 2.0])
    with pytest.raises(ValueError, match="not found"):
        fill_gaps(g)  # headcount_capped missing
    g["headcount_capped"] = g["mean_headcount_raw"]
    with pytest.raises(ValueError):
        fill_gaps(g, max_interp_bins=0)
    with pytest.raises(ValueError):
        fill_gaps(g, quiet_threshold=-0.1)


# ---------- clean_occupancy / cleaning_summary ----------

def test_clean_occupancy_end_to_end_columns_and_preservation():
    values = [2.0] * 40
    values[20] = 60.0
    values[25:30] = [np.nan] * 5
    out = clean_occupancy(_binned("A", values))
    assert list(out.columns) == [
        "floorspaceid", "timestamp", "mean_headcount_raw", "n_obs",
        "is_spike", "headcount", "fill_status",
    ]
    assert len(out) == 40
    assert out.loc[20, "mean_headcount_raw"] == 60.0 and out.loc[20, "headcount"] < 60.0
    assert out.loc[0, "headcount"] == 2.0


def test_spike_is_capped_before_interpolation():
    # The spike sits next to a short gap; interpolation must bridge to the capped value
    values = [2.0] * 30 + [80.0, np.nan, np.nan] + [2.0] * 30
    out = clean_occupancy(_binned("A", values))
    assert out.loc[31:32, "headcount"].max() < 10.0


def test_cleaning_summary_shares_sum_to_one_and_count_spikes():
    values = [2.0] * 40
    values[20] = 60.0
    values[25:30] = [np.nan] * 5
    summary = cleaning_summary(clean_occupancy(_binned("A", values)))
    row = summary.loc["A"]
    assert row["n_bins"] == 40
    assert row["n_spikes"] == 1
    assert row[[FILL_OBSERVED, FILL_INTERPOLATED, FILL_ASSUMED_EMPTY, FILL_MISSING]].sum() == pytest.approx(1.0)
    assert row["max_raw"] == 60.0


def test_cleaning_summary_missing_column_raises():
    with pytest.raises(ValueError, match="missing required columns"):
        cleaning_summary(pd.DataFrame({"floorspaceid": ["A"]}))
