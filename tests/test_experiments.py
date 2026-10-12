"""Tests for occupancy_forecast.experiments."""

import numpy as np
import pandas as pd
import pytest

from occupancy_forecast.experiments import (
    METHODS,
    ConditionResult,
    compare_conditions,
    reference_boundaries,
    run_condition,
    scaled_clean_kwargs,
)


def _binned(days=30, seed=0):
    """Two rooms, daily pattern (busy 9-17h) with noise; night bins mostly absent."""
    rng = np.random.default_rng(seed)
    parts = []
    for room in ("A", "B"):
        ts = pd.date_range("2024-03-04", periods=days * 288, freq="5min")
        hour = ts.hour + ts.minute / 60
        level = np.where((hour >= 9) & (hour < 17), 4.0, 0.0) + (room == "B")
        hc = np.clip(level + rng.normal(0, 0.4, len(ts)), 0, None)
        night = (hour < 6) | (hour >= 22)
        keep = ~(night & (rng.random(len(ts)) < 0.8))
        parts.append(pd.DataFrame({"floorspaceid": room, "timestamp": ts[keep],
                                   "mean_headcount": hc[keep], "n_obs": 3}))
    return pd.concat(parts, ignore_index=True)


FAST = dict(depths=(3, 6), leaves=(20,))


def test_scaled_clean_kwargs_keep_durations_equal():
    five = scaled_clean_kwargs(5)
    assert (five["window_bins"], five["min_periods"], five["max_interp_bins"], five["max_quiet_bins"]) == (49, 8, 3, 288)
    fifteen = scaled_clean_kwargs(15)
    assert fifteen["max_interp_bins"] * 15 == 15 and fifteen["max_quiet_bins"] * 15 == 1440
    assert fifteen["window_bins"] % 2 == 1
    one = scaled_clean_kwargs(1)
    assert one["max_interp_bins"] == 15 and one["max_quiet_bins"] == 1440
    assert scaled_clean_kwargs(5, quiet_threshold=1.0, max_quiet_bins=0)["max_quiet_bins"] == 0


@pytest.mark.parametrize("bad", [0, 16, 2.5, True, "5"])
def test_scaled_clean_kwargs_rejects_bad_bin(bad):
    with pytest.raises(ValueError):
        scaled_clean_kwargs(bad)


def test_run_condition_returns_validation_predictions_for_all_methods():
    res = run_condition(_binned(), "default", **FAST)
    assert isinstance(res, ConditionResult)
    assert res.n_train > res.n_val > 0
    assert {f"pred_{m}" for m in METHODS} <= set(res.val.columns)
    assert not res.val["pred_decision_tree"].isna().any()
    assert res.tuning["selected"].sum() == 1
    assert set(res.seconds) == {"clean", "features", "tuning"}


def test_conditions_share_the_fixed_validation_window():
    binned = _binned()
    bounds = reference_boundaries(binned)
    a = run_condition(binned, "on", boundaries=bounds, **FAST)
    b = run_condition(binned, "off", boundaries=bounds, clean_overrides={"max_quiet_bins": 0}, **FAST)
    for res in (a, b):
        for room, g in res.val.groupby("floorspaceid"):
            b1, b2 = bounds[room]
            assert g["timestamp"].min() >= b1
    # disabling the zero-fill rule removes rows, but only within the same window
    assert b.n_train < a.n_train


def test_compare_conditions_uses_identical_rows_and_observed_targets():
    binned = _binned()
    bounds = reference_boundaries(binned)
    a = run_condition(binned, "on", boundaries=bounds, **FAST)
    b = run_condition(binned, "off", boundaries=bounds, clean_overrides={"max_quiet_bins": 0}, **FAST)
    table = compare_conditions([a, b])
    assert set(table.index.get_level_values("condition")) == {"on", "off"}
    assert table["n"].nunique() == 1                       # same rows for every condition and method
    assert (table["rmse"] > 0).all() and (table["mae"] <= table["rmse"] + 1e-12).all()
    all_rows = compare_conditions([a, b], observed_only=False)
    # the synthetic data has quiet nights, so filtering to observed targets must remove rows
    assert all_rows["n"].iloc[0] > table["n"].iloc[0]
    # and the rows it keeps are exactly those whose target was measured in both conditions
    n_obs_common = len(set(map(tuple, a.val.loc[a.val["target_fill_status"] == "observed", ["floorspaceid", "timestamp"]].to_numpy()))
                       & set(map(tuple, b.val.loc[b.val["target_fill_status"] == "observed", ["floorspaceid", "timestamp"]].to_numpy())))
    assert table["n"].iloc[0] <= n_obs_common


def test_compare_conditions_validates_inputs():
    binned = _binned()
    five = run_condition(binned, "five", **FAST)
    sixty = run_condition(binned, "h60", horizon_minutes=60, **FAST)
    with pytest.raises(ValueError, match="at least one"):
        compare_conditions([])
    with pytest.raises(ValueError, match="same horizon"):
        compare_conditions([five, sixty])
    other = run_condition(binned, "other", **FAST)
    other.val = other.val.assign(timestamp=other.val["timestamp"] + pd.Timedelta(days=3650))
    with pytest.raises(ValueError, match="no validation rows"):
        compare_conditions([five, other])


def test_run_condition_raises_when_nothing_is_usable():
    # 8 bins (40 min) is shorter than the 60-min history plus 30-min horizon needed
    ts = pd.date_range("2024-03-04 09:00", periods=8, freq="5min")
    tiny = pd.DataFrame({"floorspaceid": "A", "timestamp": ts, "mean_headcount": 1.0, "n_obs": 3})
    with pytest.raises(ValueError, match="empty"):
        run_condition(tiny, "tiny", **FAST)
