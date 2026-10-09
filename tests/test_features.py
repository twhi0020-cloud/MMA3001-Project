"""Tests for occupancy_forecast.features."""

import numpy as np
import pandas as pd
import pytest

from occupancy_forecast.features import build_supervised_dataset, feature_names

T0 = pd.Timestamp("2024-03-04 09:00:00")  # a Monday


def _clean(room="A", n=40, start=T0, values=None, status=None):
    """Cleaned-style frame: headcount defaults to the bin index so shifts are checkable."""
    ts = pd.date_range(start, periods=n, freq="5min")
    hc = np.arange(n, dtype=float) if values is None else np.asarray(values, dtype=float)
    st = ["observed"] * n if status is None else status
    return pd.DataFrame({"floorspaceid": room, "timestamp": ts, "headcount": hc, "fill_status": st})


def test_target_lags_and_rolling_are_aligned():
    ds = build_supervised_dataset(_clean(n=40), horizon_minutes=30)
    row = ds[ds["timestamp"] == T0 + pd.Timedelta(minutes=5 * 20)].iloc[0]  # bin index 20
    assert row["target"] == 26            # t + 30 min = +6 bins
    assert row["lag_0m"] == 20
    assert row["lag_15m"] == 17           # -3 bins
    assert row["lag_30m"] == 14
    assert row["lag_60m"] == 8
    assert row["roll_mean_30m"] == pytest.approx(np.mean(range(15, 21)))   # last 6 bins incl. t
    assert row["roll_mean_60m"] == pytest.approx(np.mean(range(9, 21)))    # last 12 bins incl. t
    assert row["target_timestamp"] == row["timestamp"] + pd.Timedelta(minutes=30)


def test_calendar_features_describe_target_time():
    # Forecast made Sunday 23:45 for Monday 00:15: calendar features must say Monday 00:15
    start = pd.Timestamp("2024-03-03 22:00:00")  # a Sunday
    ds = build_supervised_dataset(_clean(n=60, start=start), horizon_minutes=30)
    row = ds[ds["timestamp"] == pd.Timestamp("2024-03-03 23:45:00")].iloc[0]
    assert row["target_timestamp"] == pd.Timestamp("2024-03-04 00:15:00")
    assert row["day_of_week"] == 0                       # Monday, not Sunday (6)
    assert row["hour_of_day"] == pytest.approx(0.25)
    # First row with a full 60-minute history is t = 10:00, so its target is 10:30
    first = build_supervised_dataset(_clean(n=40, start=T0), horizon_minutes=30).iloc[0]
    assert first["timestamp"] == T0 + pd.Timedelta(minutes=60)
    assert first["hour_of_day"] == pytest.approx(10.5) and first["day_of_week"] == 0


def test_features_do_not_depend_on_the_future():
    base = _clean(n=60)
    changed = base.copy()
    changed.loc[40:, "headcount"] = 999.0           # alter everything from bin 40 on
    a = build_supervised_dataset(base, horizon_minutes=30)
    b = build_supervised_dataset(changed, horizon_minutes=30)
    feats = feature_names()
    cutoff = base.loc[40, "timestamp"]
    a_past = a[a["timestamp"] < cutoff].reset_index(drop=True)
    b_past = b[b["timestamp"] < cutoff].reset_index(drop=True)
    pd.testing.assert_frame_equal(a_past[feats[:-2]], b_past[feats[:-2]])  # lag & rolling only


def test_rooms_are_independent():
    df = pd.concat([_clean("A", n=30), _clean("B", n=30, values=np.full(30, 100.0))], ignore_index=True)
    ds = build_supervised_dataset(df, horizon_minutes=30)
    a = ds[ds["floorspaceid"] == "A"]
    assert a["target"].max() < 30 and a["lag_60m"].max() < 30  # never picks up B's 100s
    assert set(ds["floorspaceid"]) == {"A", "B"}


def test_rows_with_missing_target_or_features_are_dropped():
    values = np.arange(40, dtype=float)
    values[30] = np.nan                               # a 'missing' bin
    status = ["observed"] * 40
    status[30] = "missing"
    ds = build_supervised_dataset(_clean(values=values, status=status), horizon_minutes=30)
    assert not ds[["target"] + feature_names()].isna().any().any()
    ts30 = T0 + pd.Timedelta(minutes=150)
    assert ts30 - pd.Timedelta(minutes=30) not in set(ds["timestamp"])   # its target is missing
    assert ts30 not in set(ds["timestamp"])                              # its current value is missing


def test_target_fill_status_is_carried_from_target_bin():
    status = ["observed"] * 40
    status[26] = "assumed_empty"
    ds = build_supervised_dataset(_clean(status=status), horizon_minutes=30)
    row = ds[ds["timestamp"] == T0 + pd.Timedelta(minutes=100)].iloc[0]  # bin 20 -> target bin 26
    assert row["target_fill_status"] == "assumed_empty"


def test_baseline_last_week_matches_value_one_week_before_target():
    n = 2 * 7 * 288
    ds = build_supervised_dataset(_clean(n=n), horizon_minutes=30)
    row = ds[ds["timestamp"] == T0 + pd.Timedelta(minutes=5 * 3000)].iloc[0]
    assert row["baseline_last_week"] == row["target"] - 7 * 288
    assert ds["baseline_last_week"].isna().any()      # early rows have no history


def test_invalid_arguments_and_inputs_raise():
    clean = _clean()
    with pytest.raises(ValueError, match="multiple"):
        build_supervised_dataset(clean, horizon_minutes=32)
    with pytest.raises(ValueError):
        build_supervised_dataset(clean, horizon_minutes=0)
    with pytest.raises(ValueError, match="multiple"):
        build_supervised_dataset(clean, lags_minutes=(0, 7))
    with pytest.raises(ValueError, match="missing required columns"):
        build_supervised_dataset(clean.drop(columns=["fill_status"]))
    irregular = clean.drop(index=10)
    with pytest.raises(ValueError, match="regular"):
        build_supervised_dataset(irregular)


def test_feature_names_follow_settings():
    assert feature_names((0, 10), (20,)) == ["lag_0m", "lag_10m", "roll_mean_20m", "hour_of_day", "day_of_week"]
