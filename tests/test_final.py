"""Tests for occupancy_forecast.final."""

import numpy as np
import pandas as pd
import pytest

from occupancy_forecast.features import build_supervised_dataset, feature_names
from occupancy_forecast.final import (
    FINAL_METHODS,
    block_bootstrap,
    fit_final_models,
    lookahead_flags,
    predict_methods,
)
from occupancy_forecast.validation import chronological_split, regression_metrics


def _clean(weeks=6, rooms=("A", "B"), seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for i, room in enumerate(rooms):
        ts = pd.date_range("2024-03-04", periods=weeks * 7 * 288, freq="5min")
        hour = ts.hour + ts.minute / 60
        level = np.where((hour >= 9) & (hour < 17), 4.0, 0.3) + i
        hc = np.clip(level + np.cumsum(rng.normal(0, 0.05, len(ts))) * 0 + rng.normal(0, 0.6, len(ts)), 0, None)
        parts.append(pd.DataFrame({"floorspaceid": room, "timestamp": ts, "headcount": hc, "fill_status": "observed",
                                   "is_spike": False}))
    return pd.concat(parts, ignore_index=True)


@pytest.fixture(scope="module")
def split():
    data = build_supervised_dataset(_clean(), horizon_minutes=30)
    return chronological_split(data)


def test_final_models_are_refitted_on_train_plus_validation(split):
    train, val, _ = split
    models = fit_final_models(train, val, 30)
    assert models.tree.tree_.n_node_samples[0] == len(train) + len(val)        # root sees every fitted row
    assert models.train_rows == len(train) and models.val_rows == len(val)
    full_mean = pd.concat([train, val])["target"].mean()
    assert models.seasonal._global_mean == pytest.approx(full_mean)
    assert set(models.best_params) == {"max_depth", "min_samples_leaf"}
    assert models.features == feature_names() and models.horizon_minutes == 30


def test_feature_ranges_cover_train_and_validation(split):
    train, val, _ = split
    models = fit_final_models(train, val, 30)
    full = pd.concat([train, val])
    for f, (lo, hi) in models.feature_ranges.items():
        assert lo == full[f].min() and hi == full[f].max()


def test_predict_methods_returns_every_method_aligned_with_the_input(split):
    train, val, test = split
    models = fit_final_models(train, val, 30)
    preds = predict_methods(models, test)
    assert list(preds.columns) == FINAL_METHODS and len(preds) == len(test) and preds.index.equals(test.index)
    assert np.array_equal(preds["persistence"], test["lag_0m"])
    assert not preds.drop(columns="same_time_last_week").isna().any().any()
    assert (preds[["spline_smoothing", "spline_hybrid", "decision_tree"]] >= 0).all().all()
    rmse = {m: regression_metrics(test["target"], preds[m].fillna(0))["rmse"] for m in ("decision_tree", "seasonal_average")}
    assert rmse["decision_tree"] < rmse["seasonal_average"] + 1.0                # a sane, working model


def test_final_models_validate_inputs(split):
    train, val, _ = split
    with pytest.raises(ValueError):
        fit_final_models(train.iloc[0:0], val, 30)
    with pytest.raises(ValueError):
        fit_final_models(train, val.iloc[0:0], 30)


# ---------- lookahead_flags ----------

def _flag_frame(n=40):
    ts = pd.date_range("2024-03-04", periods=n, freq="5min")
    return pd.DataFrame({"floorspaceid": "A", "timestamp": ts, "is_spike": False, "fill_status": "observed"})


def test_lookahead_flags_cover_the_following_hour_and_nothing_else():
    df = _flag_frame()
    df.loc[10, "fill_status"] = "interpolated"
    flags = lookahead_flags(df).set_index("timestamp")["uses_lookahead"]
    assert not flags.iloc[:10].any()                          # before the interpolated bin
    assert flags.iloc[10:23].all()                            # bins 10..22: window of 13 bins includes bin 10
    assert not flags.iloc[23:].any()                          # afterwards


def test_lookahead_flags_include_capped_spikes_but_not_assumed_empty():
    df = _flag_frame()
    df.loc[5, "is_spike"] = True
    df.loc[30, "fill_status"] = "assumed_empty"
    flags = lookahead_flags(df)["uses_lookahead"].to_numpy()
    assert flags[5] and flags[17] and not flags[18]
    assert not flags[30:].any()


def test_lookahead_flags_are_per_room_and_validate_inputs():
    a, b = _flag_frame(), _flag_frame().assign(floorspaceid="B")
    a.loc[3, "fill_status"] = "interpolated"
    flags = lookahead_flags(pd.concat([a, b], ignore_index=True))
    assert flags[flags["floorspaceid"] == "B"]["uses_lookahead"].sum() == 0
    with pytest.raises(ValueError, match="missing required columns"):
        lookahead_flags(a.drop(columns=["is_spike"]))
    with pytest.raises(ValueError, match="multiple"):
        lookahead_flags(a, window_minutes=62)
    with pytest.raises(ValueError):
        lookahead_flags(a, bin_minutes=0)


# ---------- block_bootstrap ----------

def _boot_data(n_blocks=60, per=24, seed=1, offset_sd=0.0):
    rng = np.random.default_rng(seed)
    blocks = np.repeat(np.arange(n_blocks), per)
    y = rng.normal(3, 1, n_blocks * per)
    shared = np.repeat(rng.normal(0, offset_sd, n_blocks), per)
    good = y + rng.normal(0, 0.3, len(y)) + shared
    bad = y + rng.normal(0, 1.0, len(y)) + shared
    return y, blocks, {"good": good, "bad": bad}


def test_bootstrap_point_estimates_match_plain_rmse_and_intervals_bracket_them():
    y, blocks, preds = _boot_data()
    table = block_bootstrap(y, preds, blocks, reference="bad", n_boot=500)
    for m in preds:
        assert table.loc[m, "rmse"] == pytest.approx(regression_metrics(y, preds[m])["rmse"])
        assert table.loc[m, "rmse_lo"] <= table.loc[m, "rmse"] <= table.loc[m, "rmse_hi"]


def test_bootstrap_detects_a_real_improvement_and_a_non_improvement():
    y, blocks, preds = _boot_data()
    table = block_bootstrap(y, preds, blocks, reference="bad", n_boot=500)
    assert table.loc["good", "improvement_lo"] > 0 and table.loc["good", "prob_better"] > 0.99
    same = block_bootstrap(y, {"ref": preds["bad"], "copy": preds["bad"].copy()}, blocks, reference="ref", n_boot=200)
    assert same.loc["copy", "improvement_pct"] == 0 and same.loc["copy", "improvement_lo"] == 0
    assert same.loc["copy", "improvement_hi"] == 0 and same.loc["copy", "prob_better"] == 0


def test_bootstrap_is_reproducible_and_seed_dependent():
    y, blocks, preds = _boot_data()
    a = block_bootstrap(y, preds, blocks, reference="bad", n_boot=300, seed=7)
    b = block_bootstrap(y, preds, blocks, reference="bad", n_boot=300, seed=7)
    c = block_bootstrap(y, preds, blocks, reference="bad", n_boot=300, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not a["rmse_lo"].equals(c["rmse_lo"])


def test_block_resampling_gives_wider_intervals_than_row_resampling_when_errors_cluster():
    y, blocks, preds = _boot_data(offset_sd=1.0)                  # errors shared within each block
    by_block = block_bootstrap(y, preds, blocks, reference="bad", n_boot=800)
    by_row = block_bootstrap(y, preds, np.arange(len(y)), reference="bad", n_boot=800)
    width = lambda t: t.loc["good", "rmse_hi"] - t.loc["good", "rmse_lo"]
    assert width(by_block) > 1.5 * width(by_row)


def test_bootstrap_validation():
    y, blocks, preds = _boot_data(n_blocks=5, per=4)
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="n_boot"):
            block_bootstrap(y, preds, blocks, reference="bad", n_boot=bad)
    for bad in (0, 1, 1.5, -0.1):
        with pytest.raises(ValueError, match="level"):
            block_bootstrap(y, preds, blocks, reference="bad", level=bad)
    with pytest.raises(ValueError, match="reference"):
        block_bootstrap(y, preds, blocks, reference="nope")
    with pytest.raises(ValueError, match="same length"):
        block_bootstrap(y, preds, blocks[:-1], reference="bad")
    with pytest.raises(ValueError, match="wrong length"):
        block_bootstrap(y, {"bad": preds["bad"][:-1]}, blocks, reference="bad")
    with pytest.raises(ValueError, match="NaN"):
        block_bootstrap(y, {"bad": np.append(preds["bad"][:-1], np.nan)}, blocks, reference="bad")
