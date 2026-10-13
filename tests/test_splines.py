"""Tests for occupancy_forecast.splines."""

import numpy as np
import pandas as pd
import pytest

from occupancy_forecast.splines import (
    SPLINE_METHODS,
    SplineAnomalyForecaster,
    SplineProfile,
)


def _frame(profile, weeks=10, rooms=("A",), noise=0.0, seed=0, horizon_min=30, slot_min=15):
    """Training-style rows: target = profile(hour of target time) + noise; lag_0m = profile(now)."""
    rng = np.random.default_rng(seed)
    parts = []
    for room in rooms:
        ts = pd.date_range("2024-03-04", periods=weeks * 7 * (1440 // slot_min), freq=f"{slot_min}min")
        tt = ts + pd.Timedelta(minutes=horizon_min)
        hour_t = (tt.hour * 60 + tt.minute) / 60.0
        hour_now = (ts.hour * 60 + ts.minute) / 60.0
        parts.append(pd.DataFrame({
            "floorspaceid": room, "timestamp": ts, "target_timestamp": tt,
            "target": profile(hour_t, tt.dayofweek) + rng.normal(0, noise, len(ts)),
            "lag_0m": profile(hour_now, ts.dayofweek),
        }))
    return pd.concat(parts, ignore_index=True)


def smooth(h, dow):
    return 3.0 + 2.0 * np.sin(2 * np.pi * (h - 6) / 24)           # peaks mid-afternoon


def step(h, dow):
    return np.where((h >= 9) & (h < 17), 5.0, 0.0)                 # abrupt day/night


def test_all_methods_recover_a_smooth_profile_between_knots():
    train = _frame(smooth, noise=0.1)
    probe = pd.DataFrame({
        "floorspaceid": "A",
        "target_timestamp": pd.to_datetime(["2024-06-05 10:07", "2024-06-05 15:41", "2024-06-05 03:22"]),
    })
    truth = smooth(np.array([10 + 7 / 60, 15 + 41 / 60, 3 + 22 / 60]), None)
    for method in SPLINE_METHODS:
        pred = SplineProfile(method).fit(train).predict(probe)
        assert np.abs(pred - truth).max() < 0.25, method


def test_profile_is_continuous_through_midnight():
    train = _frame(smooth, noise=0.05)
    model = SplineProfile("cubic").fit(train)
    just_before = pd.DataFrame({"floorspaceid": ["A"], "target_timestamp": pd.to_datetime(["2024-06-05 23:59:30"])})
    just_after = pd.DataFrame({"floorspaceid": ["A"], "target_timestamp": pd.to_datetime(["2024-06-05 00:00:30"])})
    assert abs(model.predict(just_before)[0] - model.predict(just_after)[0]) < 0.15


def test_interpolating_splines_pass_through_slot_means_and_smoothing_does_not():
    train = _frame(smooth, noise=0.8, seed=3)
    slot_mean = train.assign(slot=lambda d: (d.target_timestamp.dt.hour * 60 + d.target_timestamp.dt.minute) // 15,
                             dow=lambda d: d.target_timestamp.dt.dayofweek)
    row = slot_mean[(slot_mean.dow == 2) & (slot_mean.slot == 40)]
    centre = pd.DataFrame({"floorspaceid": ["A"], "target_timestamp": [pd.Timestamp("2024-06-05 10:07:30")]})  # 40.5*15 min
    target_mean = row["target"].mean()
    exact = SplineProfile("cubic").fit(train).evaluate(["A"], centre["target_timestamp"], clip=False)[0]
    smoothed = SplineProfile("smoothing").fit(train).evaluate(["A"], centre["target_timestamp"], clip=False)[0]
    assert exact == pytest.approx(target_mean, abs=1e-6)         # passes through the noisy mean
    assert abs(smoothed - target_mean) > 1e-3                    # smoothing does not
    truth = smooth(10.125, None)
    assert abs(smoothed - truth) < abs(exact - truth)            # and is closer to the truth


def test_pchip_does_not_overshoot_but_cubic_does_on_a_step_profile():
    train = _frame(step, noise=0.0)
    cubic = SplineProfile("cubic").fit(train).diagnostics()
    pchip = SplineProfile("pchip").fit(train).diagnostics()
    assert cubic["curve_min"].min() < -0.05          # ringing: dips below zero between zero data
    assert pchip["curve_min"].min() >= -1e-9         # shape-preserving: stays within the data range
    assert pchip["curve_max"].max() <= 5.0 + 1e-9


def test_predictions_are_never_negative():
    train = _frame(step, noise=0.0)
    probe = pd.DataFrame({"floorspaceid": "A", "target_timestamp": pd.date_range("2024-06-05", periods=96, freq="15min")})
    assert (SplineProfile("cubic").fit(train).predict(probe) >= 0).all()


def test_extrapolation_is_disabled_and_queries_stay_inside_the_domain():
    train = _frame(smooth, noise=0.05)
    model = SplineProfile("cubic").fit(train)
    curve = model._curves[("A", 2)]
    assert np.isnan(curve(np.array([-30.0, 60.0]), extrapolate=False)).all()      # outside the padded range
    every_minute = pd.DataFrame({"floorspaceid": "A",
                                 "target_timestamp": pd.date_range("2024-06-05", periods=1440, freq="min")})
    assert np.isfinite(model.predict(every_minute)).all()                          # whole day is inside


def test_unseen_day_falls_back_to_room_profile_and_unseen_room_to_global_mean():
    train = _frame(smooth, noise=0.05)
    train = train[train["target_timestamp"].dt.dayofweek != 2]                    # no Wednesdays
    model = SplineProfile("pchip").fit(train)
    wed = pd.DataFrame({"floorspaceid": ["A"], "target_timestamp": pd.to_datetime(["2024-06-05 15:00"])})
    assert model.predict(wed)[0] == pytest.approx(smooth(15.0, None), abs=0.3)
    other = pd.DataFrame({"floorspaceid": ["Z"], "target_timestamp": pd.to_datetime(["2024-06-05 15:00"])})
    assert model.predict(other)[0] == pytest.approx(train["target"].mean())


def test_weekday_weekend_grouping_keeps_the_two_profiles_apart():
    def work(h, dow):
        return np.where(dow >= 5, 0.5, 3.0 + 0 * h)
    train = _frame(work, noise=0.02)
    model = SplineProfile("pchip", day_grouping="weekday_weekend").fit(train)
    sat = pd.DataFrame({"floorspaceid": ["A"], "target_timestamp": pd.to_datetime(["2024-06-08 12:00"])})
    tue = pd.DataFrame({"floorspaceid": ["A"], "target_timestamp": pd.to_datetime(["2024-06-04 12:00"])})
    assert model.predict(sat)[0] == pytest.approx(0.5, abs=0.1)
    assert model.predict(tue)[0] == pytest.approx(3.0, abs=0.1)
    assert set(model.diagnostics()["day"].dropna()) == {0, 1}


def test_prediction_does_not_use_validation_targets():
    train = _frame(smooth, noise=0.1)
    val = _frame(smooth, weeks=1, noise=0.1, seed=9)
    model = SplineProfile("smoothing").fit(train)
    before = model.predict(val)
    after = model.predict(val.assign(target=999.0))
    assert np.array_equal(before, after)


def test_profile_validation_and_state_errors():
    with pytest.raises(ValueError):
        SplineProfile("quadratic")
    with pytest.raises(ValueError):
        SplineProfile(slot_minutes=7)
    with pytest.raises(ValueError):
        SplineProfile(day_grouping="month")
    with pytest.raises(ValueError):
        SplineProfile(min_knots=3)
    train = _frame(smooth, weeks=2)
    with pytest.raises(RuntimeError):
        SplineProfile().predict(train)
    with pytest.raises(RuntimeError):
        SplineProfile().diagnostics()
    with pytest.raises(ValueError, match="empty"):
        SplineProfile().fit(train.iloc[0:0])
    with pytest.raises(ValueError, match="missing required columns"):
        SplineProfile().fit(train.drop(columns=["target"]))


def test_hybrid_recovers_the_persistence_fraction():
    # target deviates from the profile by exactly 0.5 * (current deviation)
    rng = np.random.default_rng(1)
    base = _frame(smooth, noise=0.0)
    now_dev = rng.normal(0, 1.5, len(base))
    base["lag_0m"] = smooth((base["timestamp"].dt.hour * 60 + base["timestamp"].dt.minute) / 60.0, None) + now_dev
    base["target"] = smooth((base["target_timestamp"].dt.hour * 60 + base["target_timestamp"].dt.minute) / 60.0, None) + 0.5 * now_dev
    model = SplineAnomalyForecaster(method="smoothing").fit(base)
    assert model.phi_ == pytest.approx(0.5, abs=0.05)
    pred = model.predict(base)
    assert np.sqrt(np.mean((pred - base["target"]) ** 2)) < 0.25


def test_hybrid_with_fully_persistent_deviations_gives_phi_near_one():
    # The same deviation d is present now and at the target time -> phi = 1. A noisy fitted
    # profile biases phi towards 0 (regression dilution), so use many weeks to make it small.
    rng = np.random.default_rng(2)
    df = _frame(smooth, weeks=80, noise=0.0)
    d = rng.normal(0, 1.0, len(df))
    hour = lambda t: (t.dt.hour * 60 + t.dt.minute) / 60.0
    df["lag_0m"] = smooth(hour(df["timestamp"]), None) + d
    df["target"] = smooth(hour(df["target_timestamp"]), None) + d
    model = SplineAnomalyForecaster(method="pchip").fit(df)
    assert model.phi_ == pytest.approx(1.0, abs=0.05)
    assert (model.predict(df) >= 0).all()


def test_hybrid_with_no_persistence_gives_phi_near_zero():
    rng = np.random.default_rng(3)
    df = _frame(smooth, noise=0.0)
    hour = lambda t: (t.dt.hour * 60 + t.dt.minute) / 60.0
    df["lag_0m"] = smooth(hour(df["timestamp"]), None) + rng.normal(0, 1.0, len(df))
    df["target"] = smooth(hour(df["target_timestamp"]), None) + rng.normal(0, 1.0, len(df))
    assert abs(SplineAnomalyForecaster(method="pchip").fit(df).phi_) < 0.05


def test_hybrid_errors():
    df = _frame(smooth, weeks=2)
    with pytest.raises(RuntimeError):
        SplineAnomalyForecaster().predict(df)
    with pytest.raises(ValueError, match="empty"):
        SplineAnomalyForecaster().fit(df.iloc[0:0])
    with pytest.raises(ValueError, match="missing required columns"):
        SplineAnomalyForecaster().fit(df.drop(columns=["lag_0m"]))
    fitted = SplineAnomalyForecaster().fit(df)
    with pytest.raises(ValueError, match="missing required columns"):
        fitted.predict(df.drop(columns=["timestamp"]))


# ---------- padding and profile reuse ----------

def test_short_wrap_matches_full_wrap_for_a_smooth_profile_and_stays_continuous_at_midnight():
    train = _frame(smooth, noise=0.3, seed=5)
    probe = pd.DataFrame({"floorspaceid": "A",
                          "target_timestamp": pd.date_range("2024-06-05", periods=96, freq="15min")})
    full = SplineProfile("smoothing", pad_hours=24).fit(train).predict(probe)
    short = SplineProfile("smoothing", pad_hours=3).fit(train).predict(probe)
    assert 1e-9 < np.abs(full - short).max() < 0.1       # differs a little (so pad_hours is used), but only a little
    edge = pd.DataFrame({"floorspaceid": "A",
                         "target_timestamp": pd.to_datetime(["2024-06-05 23:59:30", "2024-06-05 00:00:30"])})
    both = SplineProfile("cubic", pad_hours=3).fit(train).predict(edge)
    assert abs(both[0] - both[1]) < 0.2
    every_minute = pd.DataFrame({"floorspaceid": "A", "target_timestamp": pd.date_range("2024-06-05", periods=1440, freq="min")})
    assert np.isfinite(SplineProfile("pchip", pad_hours=0.25).fit(train).predict(every_minute)).all()


def test_default_wrap_is_the_full_day():
    train = _frame(smooth, noise=0.2, seed=6)
    a = SplineProfile("cubic").fit(train)
    b = SplineProfile("cubic", pad_hours=24.0).fit(train)
    probe = pd.DataFrame({"floorspaceid": "A", "target_timestamp": pd.date_range("2024-06-05", periods=96, freq="15min")})
    assert np.array_equal(a.predict(probe), b.predict(probe))


@pytest.mark.parametrize("bad", [0, -1, 24.5, True, "3"])
def test_pad_hours_validation(bad):
    with pytest.raises(ValueError, match="pad_hours"):
        SplineProfile(pad_hours=bad)


def test_hybrid_reusing_a_fitted_profile_gives_identical_results():
    rng = np.random.default_rng(4)
    df = _frame(smooth, weeks=20, noise=0.2)
    df["lag_0m"] = df["lag_0m"] + rng.normal(0, 1, len(df))
    separate = SplineAnomalyForecaster(method="pchip").fit(df)
    shared_profile = SplineProfile("pchip").fit(df)
    shared = SplineAnomalyForecaster(method="pchip").fit(df, profile=shared_profile)
    assert shared.profile is shared_profile
    assert shared.phi_ == pytest.approx(separate.phi_)
    assert np.array_equal(shared.predict(df), separate.predict(df))
    with pytest.raises(RuntimeError, match="fitted"):
        SplineAnomalyForecaster().fit(df, profile=SplineProfile("pchip"))
