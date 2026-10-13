"""Tests for occupancy_forecast.models."""

import numpy as np
import pandas as pd
import pytest

from occupancy_forecast.models import (
    SeasonalAverage,
    fit_decision_tree,
    fit_linear,
    permutation_importance_table,
    predict_last_week,
    predict_persistence,
    tune_decision_tree,
    tune_decision_tree_fast,
)

FEATURES = ["hour_of_day", "lag_0m"]


def _seasonal_df(n=600, seed=0, noise=0.1):
    """Headcount = f(hour of day) + small noise; lag_0m is pure noise."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-03-04", periods=n, freq="30min")
    hour = ts.hour + ts.minute / 60
    target = np.where((hour >= 9) & (hour < 17), 5.0, 0.5) + rng.normal(0, noise, n)
    return pd.DataFrame({
        "floorspaceid": "A", "timestamp": ts, "target_timestamp": ts,
        "hour_of_day": hour, "day_of_week": ts.dayofweek,
        "lag_0m": rng.normal(2, 1, n), "baseline_last_week": np.roll(target, 336),
        "target": target,
    })


def test_persistence_and_last_week_return_the_expected_columns():
    df = _seasonal_df(50)
    assert np.array_equal(predict_persistence(df), df["lag_0m"].to_numpy())
    assert np.array_equal(predict_last_week(df), df["baseline_last_week"].to_numpy())
    with pytest.raises(ValueError):
        predict_persistence(df.drop(columns=["lag_0m"]))
    with pytest.raises(ValueError):
        predict_last_week(df.drop(columns=["baseline_last_week"]))


def test_seasonal_average_learns_group_means():
    ts = pd.to_datetime(["2024-03-04 10:00", "2024-03-11 10:00", "2024-03-04 22:00"])  # Mondays
    train = pd.DataFrame({"floorspaceid": "A", "target_timestamp": ts, "target": [4.0, 6.0, 0.0]})
    model = SeasonalAverage().fit(train)
    test = pd.DataFrame({
        "floorspaceid": ["A", "A"],
        "target_timestamp": pd.to_datetime(["2024-03-18 10:05", "2024-03-18 22:00"]),  # same slots
    })
    assert model.predict(test).tolist() == pytest.approx([5.0, 0.0])


def test_seasonal_average_fallbacks_and_errors():
    train = pd.DataFrame({
        "floorspaceid": "A",
        "target_timestamp": pd.to_datetime(["2024-03-04 10:00", "2024-03-05 10:00"]),  # Mon, Tue
        "target": [2.0, 4.0],
    })
    model = SeasonalAverage().fit(train)
    wednesday = pd.DataFrame({"floorspaceid": ["A"], "target_timestamp": pd.to_datetime(["2024-03-06 10:00"])})
    assert model.predict(wednesday)[0] == pytest.approx(3.0)       # (room, slot) fallback
    other_room = pd.DataFrame({"floorspaceid": ["Z"], "target_timestamp": pd.to_datetime(["2024-03-06 03:00"])})
    assert model.predict(other_room)[0] == pytest.approx(3.0)      # global mean fallback
    with pytest.raises(RuntimeError):
        SeasonalAverage().predict(wednesday)
    with pytest.raises(ValueError):
        SeasonalAverage(slot_minutes=7)
    with pytest.raises(ValueError):
        SeasonalAverage().fit(train.iloc[0:0])


def test_linear_fit_recovers_exact_relationship():
    df = pd.DataFrame({"x": np.arange(20.0)})
    df["target"] = 3 * df["x"] + 1
    model = fit_linear(df, ["x"])
    assert model.predict(pd.DataFrame({"x": [10.0]}))[0] == pytest.approx(31.0)
    with pytest.raises(ValueError):
        fit_linear(df.iloc[0:0], ["x"])


def test_tuning_returns_table_and_best_has_lowest_validation_rmse():
    train, val = _seasonal_df(800, seed=1), _seasonal_df(300, seed=2)
    results, best = tune_decision_tree(train, val, FEATURES, [1, 2, 4, None], [1, 20], tolerance=0.0)
    assert {"max_depth", "min_samples_leaf", "n_leaves", "train_rmse", "val_rmse", "val_mae", "fit_seconds"} <= set(results.columns)
    assert len(results) == 8
    chosen = results[(results["max_depth"].fillna(-1) == (-1 if best["max_depth"] is None else best["max_depth"]))
                     & (results["min_samples_leaf"] == best["min_samples_leaf"])].iloc[0]
    assert chosen["val_rmse"] == pytest.approx(results["val_rmse"].min())
    assert chosen["val_rmse"] < 0.5                    # the day/night step is learnable


def test_deeper_trees_never_have_higher_training_error():
    train, val = _seasonal_df(800, seed=1), _seasonal_df(300, seed=2)
    results, _ = tune_decision_tree(train, val, FEATURES, [1, 3, 8], [1])
    r = results.sort_values("max_depth")["train_rmse"].to_numpy()
    assert (np.diff(r) <= 1e-9).all()


def test_tuning_validates_inputs():
    df = _seasonal_df(50)
    with pytest.raises(ValueError):
        tune_decision_tree(df, df, FEATURES, [], [1])
    with pytest.raises(ValueError):
        tune_decision_tree(df.iloc[0:0], df, FEATURES, [2], [1])


def test_fit_decision_tree_respects_settings_and_is_repeatable():
    df = _seasonal_df(300)
    a = fit_decision_tree(df, FEATURES, 3, 5)
    b = fit_decision_tree(df, FEATURES, 3, 5)
    assert a.get_depth() <= 3
    assert np.array_equal(a.predict(df[FEATURES]), b.predict(df[FEATURES]))
    with pytest.raises(ValueError):
        fit_decision_tree(df.iloc[0:0], FEATURES, 3, 5)


def test_permutation_importance_ranks_the_informative_feature_first():
    train, val = _seasonal_df(800, seed=1), _seasonal_df(400, seed=2)
    model = fit_decision_tree(train, FEATURES, 4, 5)
    table = permutation_importance_table(model, val, FEATURES)
    assert table.iloc[0]["feature"] == "hour_of_day"
    assert table.iloc[0]["rmse_increase"] > 1.0
    assert list(table.columns) == ["feature", "rmse_increase", "rmse_increase_std"]
    with pytest.raises(ValueError):
        permutation_importance_table(model, val.iloc[0:0], FEATURES)


def test_tolerance_picks_fewest_leaves_within_noise_of_the_best():
    train, val = _seasonal_df(800, seed=1), _seasonal_df(300, seed=2)
    strict, _ = tune_decision_tree(train, val, FEATURES, [1, 3, 6, None], [1, 5], tolerance=0.0)
    loose, best = tune_decision_tree(train, val, FEATURES, [1, 3, 6, None], [1, 5], tolerance=0.05)
    pick_strict = strict[strict["selected"]].iloc[0]
    pick_loose = loose[loose["selected"]].iloc[0]
    assert strict["selected"].sum() == 1 and loose["selected"].sum() == 1
    assert pick_strict["val_rmse"] == pytest.approx(strict["val_rmse"].min())
    assert pick_loose["n_leaves"] <= pick_strict["n_leaves"]
    assert pick_loose["val_rmse"] <= loose["val_rmse"].min() * 1.05 + 1e-12
    eligible = loose[loose["val_rmse"] <= loose["val_rmse"].min() * 1.05]
    assert pick_loose["n_leaves"] == eligible["n_leaves"].min()


def test_negative_tolerance_rejected():
    df = _seasonal_df(60)
    with pytest.raises(ValueError, match="tolerance"):
        tune_decision_tree(df, df, FEATURES, [2], [1], tolerance=-0.1)


# ---------- tune_decision_tree_fast ----------

def test_fast_tuning_structure_and_no_duplicate_fits():
    train, val = _seasonal_df(1200, seed=1), _seasonal_df(300, seed=2)
    results, best = tune_decision_tree_fast(train, val, FEATURES, stride=2)
    assert {"stage", "selected", "val_rmse", "n_leaves"} <= set(results.columns)
    assert results["selected"].sum() == 1
    assert set(results["stage"]) == {1, 2}
    assert not results.duplicated(["max_depth", "min_samples_leaf"]).any()
    assert len(results[results["stage"] == 1]) == 6
    assert len(results) <= 6 + 9
    assert set(best) == {"max_depth", "min_samples_leaf"}


def test_fast_tuning_stage_two_is_a_neighbourhood_of_the_stage_one_winner():
    train, val = _seasonal_df(1200, seed=1), _seasonal_df(300, seed=2)
    results, _ = tune_decision_tree_fast(train, val, FEATURES, stride=1)
    stage1, stage2 = results[results["stage"] == 1], results[results["stage"] == 2]
    row = stage1.sort_values(["val_rmse"]).iloc[0]
    eligible = stage1[stage1["val_rmse"] <= stage1["val_rmse"].min() * 1.005]
    win = eligible.sort_values(["n_leaves", "val_rmse"]).iloc[0]
    d0, l0 = int(win["max_depth"]), int(win["min_samples_leaf"])
    assert set(stage2["max_depth"].astype(int)) <= {max(2, d0 - 2), d0, d0 + 2}
    assert set(stage2["min_samples_leaf"].astype(int)) <= {max(1, l0 // 4), l0, l0 * 4}


def test_fast_tuning_selected_is_simplest_within_tolerance():
    train, val = _seasonal_df(1200, seed=1), _seasonal_df(300, seed=2)
    results, _ = tune_decision_tree_fast(train, val, FEATURES, stride=1, tolerance=0.05)
    eligible = results[results["val_rmse"] <= results["val_rmse"].min() * 1.05]
    assert results[results["selected"]].iloc[0]["n_leaves"] == eligible["n_leaves"].min()


def test_fast_tuning_uses_only_every_stride_th_training_row():
    train, val = _seasonal_df(1200, seed=1), _seasonal_df(300, seed=2)
    poisoned = train.copy()
    poisoned.loc[poisoned.index % 2 == 1, "target"] = 1e6          # odd rows are garbage
    results, _ = tune_decision_tree_fast(poisoned, val, FEATURES, stride=2)
    assert results["train_rmse"].max() < 5                          # garbage rows were never used
    results_all, _ = tune_decision_tree_fast(poisoned, val, FEATURES, stride=1)
    assert results_all["train_rmse"].max() > 1000                   # they are used when stride=1


def test_fast_tuning_validates_inputs():
    df = _seasonal_df(100)
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError, match="stride"):
            tune_decision_tree_fast(df, df, FEATURES, stride=bad)
    with pytest.raises(ValueError):
        tune_decision_tree_fast(df, df, FEATURES, coarse_depths=())
    with pytest.raises(ValueError, match="positive integers"):
        tune_decision_tree_fast(df, df, FEATURES, coarse_leaves=(10, 0))
    with pytest.raises(ValueError):
        tune_decision_tree_fast(df.iloc[0:0], df, FEATURES)
    with pytest.raises(ValueError, match="tolerance"):
        tune_decision_tree_fast(df, df, FEATURES, tolerance=-1)
