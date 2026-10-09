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
    results, best = tune_decision_tree(train, val, FEATURES, [1, 2, 4, None], [1, 20])
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
