"""Tests for occupancy_forecast.forecast."""

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from occupancy_forecast.cleaning import clean_occupancy
from occupancy_forecast.data_loading import bin_readings
from occupancy_forecast.features import build_supervised_dataset, feature_names
from occupancy_forecast.forecast import (
    OUTPUT_COLUMNS,
    forecast_headcount,
    load_bundle,
    save_bundle,
)

START = pd.Timestamp("2024-03-04 08:00:00")              # Monday; AEDT, UTC+11


def _raw(room="A", minutes=480, start=START, base=2.0, seed=0, offset="+11"):
    """One reading per minute; headcount follows a gentle daily-like wave plus small noise."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=minutes, freq="min")
    hc = np.clip(base + 1.5 * np.sin(np.arange(minutes) / 60.0) + rng.normal(0, 0.15, minutes), 0, None)
    return pd.DataFrame({"floorspaceid": room, "headcount": hc,
                         "collecteddate": [t.strftime("%Y-%m-%d %H:%M:%S") + offset for t in ts]})


def _fitted_bundle(raw):
    clean = clean_occupancy(bin_readings(raw, 5))
    data = build_supervised_dataset(clean, horizon_minutes=30)
    tree = DecisionTreeRegressor(max_depth=4, random_state=0).fit(data[feature_names()], data["target"])
    ranges = {f: (float(data[f].min()), float(data[f].max())) for f in feature_names()}
    bundle = {"model": tree, "features": feature_names(), "horizon_minutes": 30, "bin_minutes": 5,
              "lags_minutes": (0, 15, 30, 60), "rolling_minutes": (30, 60), "feature_ranges": ranges,
              "params": {"max_depth": 4, "min_samples_leaf": 1}}
    return bundle, data, tree


@pytest.fixture(scope="module")
def setup():
    raw = pd.concat([_raw("A", seed=1), _raw("B", seed=2, base=3.0)], ignore_index=True)
    bundle, data, tree = _fitted_bundle(raw)
    return raw, bundle, data, tree


def _local(raw):
    return pd.to_datetime(raw["collecteddate"], utc=True).dt.tz_convert("Australia/Melbourne").dt.tz_localize(None)


def test_live_forecast_equals_the_trained_tree_on_the_matching_training_row(setup):
    raw, bundle, data, tree = setup
    t0 = START + pd.Timedelta(minutes=5 * 70)
    history = raw[_local(raw) < t0 + pd.Timedelta(minutes=5)]          # only data up to the end of bin t0
    out = forecast_headcount(history, bundle).set_index("floorspaceid")
    row = data[data["timestamp"] == t0].set_index("floorspaceid")
    for room in ("A", "B"):
        expected = max(0.0, tree.predict(row.loc[[room], feature_names()])[0])
        assert out.loc[room, "predicted_headcount"] == pytest.approx(expected)
        assert out.loc[room, "status"] == "ok" and out.loc[room, "as_of"] == t0
        assert out.loc[room, "forecast_for"] == t0 + pd.Timedelta(minutes=30)


def test_readings_after_as_of_are_ignored(setup):
    raw, bundle, _, _ = setup
    t0 = START + pd.Timedelta(minutes=5 * 70)
    truncated = raw[_local(raw) < t0 + pd.Timedelta(minutes=5)]
    a = forecast_headcount(truncated, bundle)
    b = forecast_headcount(raw, bundle, as_of=t0)                      # later data present but must be ignored
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_as_of_defaults_to_the_latest_bin_and_accepts_aware_or_naive_times(setup):
    raw, bundle, _, _ = setup
    latest = forecast_headcount(raw, bundle)
    assert (latest["as_of"] == _local(raw).max().floor("5min")).all()
    naive = forecast_headcount(raw, bundle, as_of="2024-03-04 12:00:00")
    aware = forecast_headcount(raw, bundle, as_of="2024-03-04 12:00:00+11:00")
    pd.testing.assert_frame_equal(naive, aware)


def test_output_schema_and_non_negative_predictions(setup):
    raw, bundle, _, _ = setup
    out = forecast_headcount(raw, bundle)
    assert list(out.columns) == OUTPUT_COLUMNS and len(out) == 2 and set(out["floorspaceid"]) == {"A", "B"}
    assert (out["predicted_headcount"] >= 0).all() and out["status"].eq("ok").all()
    assert out.attrs["dropped_readings"] == 0


def test_short_history_gives_insufficient_history_not_a_guess(setup):
    raw, bundle, _, _ = setup
    out = forecast_headcount(raw[_local(raw) < START + pd.Timedelta(minutes=30)], bundle)
    assert out["status"].eq("insufficient_history").all()
    assert out["predicted_headcount"].isna().all() and not out["outside_training_range"].any()


def test_a_room_that_went_quiet_after_a_low_count_is_treated_as_empty(setup):
    _, bundle, _, _ = setup
    quiet = _raw("A", minutes=120, base=0.0, seed=3)
    quiet["headcount"] = 0.0                                           # empty, then the sensor goes silent
    out = forecast_headcount(quiet, bundle, as_of=START + pd.Timedelta(minutes=120 + 90))
    assert out.loc[0, "status"] == "ok" and out.loc[0, "current_headcount"] == 0.0


def test_a_room_that_went_quiet_while_busy_is_not_guessed(setup):
    _, bundle, _, _ = setup
    busy = _raw("A", minutes=120, base=4.0, seed=3)
    busy["headcount"] = 4.0
    out = forecast_headcount(busy, bundle, as_of=START + pd.Timedelta(minutes=120 + 90))
    assert out.loc[0, "status"] == "insufficient_history" and np.isnan(out.loc[0, "predicted_headcount"])


def test_values_outside_the_training_range_are_flagged(setup):
    raw, bundle, _, _ = setup
    crowded = raw[raw["floorspaceid"] == "A"].copy()
    crowded.loc[crowded.index[-120:], "headcount"] = 60.0
    out = forecast_headcount(crowded, bundle)
    assert out.loc[0, "outside_training_range"] and out.loc[0, "status"] == "ok"
    # control: a time from the middle of the history lies inside the range the tree was trained on
    # (the very last bins can legitimately fall outside it, since training rows stop 30 min earlier)
    mid = forecast_headcount(raw[raw["floorspaceid"] == "A"], bundle, as_of=START + pd.Timedelta(minutes=5 * 70))
    assert not mid.loc[0, "outside_training_range"]


def test_missing_headcounts_are_dropped_and_counted(setup):
    raw, bundle, _, _ = setup
    messy = raw.copy()
    messy["headcount"] = messy["headcount"].astype(object)
    messy.loc[messy.index[:7], "headcount"] = None
    messy.loc[messy.index[7], "headcount"] = "n/a"
    out = forecast_headcount(messy, bundle)
    assert out.attrs["dropped_readings"] == 8 and len(out) == 2


def test_invalid_input_raises(setup):
    raw, bundle, _, _ = setup
    with pytest.raises(ValueError, match="missing required columns"):
        forecast_headcount(raw.drop(columns=["headcount"]), bundle)
    with pytest.raises(ValueError, match="empty"):
        forecast_headcount(raw.iloc[0:0], bundle)
    with pytest.raises(ValueError, match="negative"):
        forecast_headcount(raw.assign(headcount=-1.0), bundle)
    with pytest.raises(ValueError, match="UTC offset"):
        forecast_headcount(raw.assign(collecteddate=raw["collecteddate"].str[:-3]), bundle)
    with pytest.raises(ValueError, match="no usable readings"):
        forecast_headcount(raw.assign(headcount=np.nan), bundle)
    with pytest.raises(ValueError, match="no readings at or before"):
        forecast_headcount(raw, bundle, as_of="2020-01-01 00:00:00")
    with pytest.raises(ValueError, match="DataFrame"):
        forecast_headcount([1, 2, 3], bundle)


def test_bundle_round_trip_and_validation(setup, tmp_path):
    raw, bundle, _, _ = setup
    path = save_bundle(bundle, tmp_path / "models" / "bundle.joblib")
    loaded = load_bundle(path)
    pd.testing.assert_frame_equal(forecast_headcount(raw, bundle), forecast_headcount(raw, str(path)))
    assert loaded["horizon_minutes"] == 30 and loaded["features"] == bundle["features"]
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "nope.joblib")
    with pytest.raises(ValueError, match="missing keys"):
        forecast_headcount(raw, {k: v for k, v in bundle.items() if k != "model"})
    with pytest.raises(ValueError, match="do not match"):
        save_bundle({**bundle, "features": bundle["features"][:-1]}, tmp_path / "bad.joblib")
    with pytest.raises(ValueError, match="bundle dict"):
        forecast_headcount(raw, 123)
