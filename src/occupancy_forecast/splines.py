"""Smooth time-of-day profiles of headcount, built with the splines taught in the unit.

The seasonal-average baseline (see :class:`occupancy_forecast.models.SeasonalAverage`) is a
*step* function: every 15-minute slot has its own mean. Here the same slot means are turned
into a *smooth* curve of headcount against time of day, one curve per room and day type.
Five constructions from the unit's spline notes are compared:

* ``linear`` - straight lines between slot means (still continuous, but slope jumps);
* ``cubic`` - piecewise cubics with continuous slope and curvature (``not-a-knot`` ends);
* ``pchip`` - shape-preserving cubic Hermite: no new peaks or dips between slot means;
* ``akima`` - smooth cubic from local slopes;
* ``smoothing`` - a smoothing spline that does *not* pass through the noisy slot means; the
  smoothing strength is chosen by generalised cross-validation (GCV).

Interpolating splines pass through every slot mean, so they reproduce its noise; the
smoothing spline trades closeness for smoothness. A headcount cannot be negative, so
predictions are clipped at zero; how often a curve would have gone negative on its own is
reported by :meth:`SplineProfile.diagnostics` (the shape-preservation point from the notes).

**Midnight and extrapolation.** Time of day is circular, so the 24 h profile is repeated one
day on each side before fitting. The curve is then smooth through midnight and every query
lies *inside* the data range. Evaluation uses ``extrapolate=False``, so a query outside the
range would return ``NaN`` (and raise here) instead of a silent extrapolated value.

**Hybrid forecast.** A profile alone ignores what is happening now, and the earlier results
show recent history matters most. :class:`SplineAnomalyForecaster` therefore predicts
``profile(t+h) + phi * (current - profile(t))``: the expected level at the target time plus a
fraction ``phi`` (fitted by least squares on training data) of today's deviation from normal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import (
    Akima1DInterpolator,
    CubicSpline,
    PchipInterpolator,
    make_interp_spline,
    make_smoothing_spline,
)

SPLINE_METHODS = ("linear", "cubic", "pchip", "akima", "smoothing")
DAY_GROUPINGS = ("dow", "weekday_weekend")
_MIN_KNOTS_FLOOR = 5  # make_smoothing_spline needs at least 5 points


def _fractional_hours(ts: pd.Series) -> np.ndarray:
    return ((ts.dt.hour * 60 + ts.dt.minute + ts.dt.second / 60.0) / 60.0).to_numpy(dtype=float)


def _day_key(ts: pd.Series, grouping: str) -> np.ndarray:
    dow = ts.dt.dayofweek.to_numpy()
    return dow if grouping == "dow" else (dow >= 5).astype(int)


def _build_interpolant(method: str, x: np.ndarray, y: np.ndarray, w: np.ndarray):
    if method == "linear":
        return make_interp_spline(x, y, k=1)
    if method == "cubic":
        return CubicSpline(x, y, bc_type="not-a-knot")
    if method == "pchip":
        return PchipInterpolator(x, y)
    if method == "akima":
        return Akima1DInterpolator(x, y)
    return make_smoothing_spline(x, y, w=w, lam=None)  # lam=None -> GCV


def _fit_curve(method: str, slot_hours: np.ndarray, means: np.ndarray, counts: np.ndarray):
    """Fit one periodic profile: repeat the 24 h data one day either side, then interpolate."""
    x = np.concatenate([slot_hours - 24.0, slot_hours, slot_hours + 24.0])
    y = np.tile(means, 3)
    w = np.tile(counts.astype(float), 3)
    return _build_interpolant(method, x, y, w)


class SplineProfile:
    """Smooth mean headcount as a function of time of day, per room and day type.

    Args:
        method: One of ``SPLINE_METHODS``.
        slot_minutes: Width of the time-of-day slots whose means the spline is fitted to
            (positive divisor of 1440).
        day_grouping: ``"dow"`` (a profile per day of week) or ``"weekday_weekend"``
            (two profiles per room, with more data per slot).
        min_knots: Minimum number of populated slots needed to fit a curve for a group
            (at least 5). Smaller groups fall back to the room-level profile.

    Raises:
        ValueError: If an argument is invalid.
    """

    def __init__(self, method: str = "smoothing", slot_minutes: int = 15,
                 day_grouping: str = "dow", min_knots: int = 8):
        if method not in SPLINE_METHODS:
            raise ValueError(f"method must be one of {SPLINE_METHODS}, got {method!r}")
        if isinstance(slot_minutes, bool) or not isinstance(slot_minutes, int) \
                or slot_minutes <= 0 or 1440 % slot_minutes:
            raise ValueError(f"slot_minutes must be a positive divisor of 1440, got {slot_minutes!r}")
        if day_grouping not in DAY_GROUPINGS:
            raise ValueError(f"day_grouping must be one of {DAY_GROUPINGS}, got {day_grouping!r}")
        if isinstance(min_knots, bool) or not isinstance(min_knots, int) or min_knots < _MIN_KNOTS_FLOOR:
            raise ValueError(f"min_knots must be an integer >= {_MIN_KNOTS_FLOOR}, got {min_knots!r}")
        self.method = method
        self.slot_minutes = slot_minutes
        self.day_grouping = day_grouping
        self.min_knots = min_knots
        self._curves: dict = {}
        self._n_knots: dict = {}
        self._global_mean: float | None = None

    def _slot_table(self, rooms, timestamps, values) -> pd.DataFrame:
        ts = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
        minutes = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
        return pd.DataFrame({
            "room": np.asarray(rooms),
            "day": _day_key(ts, self.day_grouping),
            "slot": minutes // self.slot_minutes,
            "y": np.asarray(values, dtype=float),
        })

    def _fit_group(self, key, frame: pd.DataFrame) -> None:
        stats = frame.groupby("slot")["y"].agg(["mean", "count"]).sort_index()
        if len(stats) < self.min_knots:
            return
        centres = (stats.index.to_numpy() + 0.5) * self.slot_minutes / 60.0
        self._curves[key] = _fit_curve(self.method, centres, stats["mean"].to_numpy(), stats["count"].to_numpy())
        self._n_knots[key] = len(stats)

    def fit(self, train: pd.DataFrame) -> "SplineProfile":
        """Fit the profiles from training data only.

        Args:
            train: Rows with ``floorspaceid``, ``target_timestamp`` and ``target`` (people).

        Returns:
            ``self``.

        Raises:
            ValueError: If ``train`` is empty or required columns are missing.
        """
        needed = {"floorspaceid", "target_timestamp", "target"}
        if needed - set(train.columns):
            raise ValueError(f"train is missing required columns: {sorted(needed - set(train.columns))}")
        if train.empty:
            raise ValueError("cannot fit on an empty training set")

        table = self._slot_table(train["floorspaceid"], train["target_timestamp"], train["target"])
        self._global_mean = float(table["y"].mean())
        self._curves, self._n_knots = {}, {}
        for (room, day), g in table.groupby(["room", "day"]):
            self._fit_group((room, day), g)
        for room, g in table.groupby("room"):  # room-level fallback across all days
            self._fit_group((room, None), g)
        return self

    def evaluate(self, rooms, timestamps, clip: bool = True) -> np.ndarray:
        """Evaluate the fitted profile at given rooms and times of day.

        Args:
            rooms: Room identifier per query.
            timestamps: Query times (only the time of day and day type are used).
            clip: Clip negative values to zero (headcount cannot be negative).

        Returns:
            Expected headcount (people) per query. Groups without a curve fall back to the
            room-level curve, then to the overall training mean.

        Raises:
            RuntimeError: If called before :meth:`fit`, or a query falls outside the
                curve's domain (extrapolation is disabled).
        """
        if self._global_mean is None:
            raise RuntimeError("SplineProfile must be fitted before evaluate")
        ts = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)
        rooms = np.asarray(rooms)
        hours = _fractional_hours(ts)
        days = _day_key(ts, self.day_grouping)
        out = np.full(len(ts), self._global_mean, dtype=float)

        keys = pd.DataFrame({"room": rooms, "day": days})
        for (room, day), idx in keys.groupby(["room", "day"]).groups.items():
            curve = self._curves.get((room, day), self._curves.get((room, None)))
            if curve is None:
                continue
            idx = np.asarray(idx)
            vals = curve(hours[idx], extrapolate=False)
            if np.isnan(vals).any():
                raise RuntimeError("spline queried outside its domain; extrapolation is disabled")
            out[idx] = vals
        return np.clip(out, 0.0, None) if clip else out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict the headcount at each row's target time from the fitted profile.

        Args:
            df: Rows with ``floorspaceid`` and ``target_timestamp``.

        Returns:
            Predicted headcount (people), never negative.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        return self.evaluate(df["floorspaceid"], df["target_timestamp"])

    def diagnostics(self) -> pd.DataFrame:
        """Summarise each fitted curve, including whether it would go negative.

        Returns:
            DataFrame with one row per fitted curve: ``room``, ``day`` (``None`` = all days),
            ``n_knots``, ``curve_min`` and ``curve_max`` (people, before clipping) over a
            dense grid of the day.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._global_mean is None:
            raise RuntimeError("SplineProfile must be fitted before diagnostics")
        grid = np.linspace(0.0, 24.0, 2401, endpoint=False)
        rows = []
        for key, curve in self._curves.items():
            vals = curve(grid, extrapolate=False)
            rows.append({"room": key[0], "day": key[1], "n_knots": self._n_knots[key],
                         "curve_min": float(np.min(vals)), "curve_max": float(np.max(vals))})
        return pd.DataFrame(rows, columns=["room", "day", "n_knots", "curve_min", "curve_max"])


class SplineAnomalyForecaster:
    """Profile at the target time plus a fraction of today's deviation from the profile.

    ``prediction = profile(t + h) + phi * (current - profile(t))`` where ``phi`` is fitted by
    least squares through the origin on training data. ``phi`` near 1 means "today's
    deviation persists"; near 0 means "it fades back to the normal profile".

    Args:
        **profile_kwargs: Passed to :class:`SplineProfile`.
    """

    def __init__(self, **profile_kwargs):
        self.profile = SplineProfile(**profile_kwargs)
        self.phi_: float | None = None

    @staticmethod
    def _check(df: pd.DataFrame, extra: set) -> None:
        needed = {"floorspaceid", "timestamp", "target_timestamp", "lag_0m"} | extra
        if needed - set(df.columns):
            raise ValueError(f"df is missing required columns: {sorted(needed - set(df.columns))}")

    def fit(self, train: pd.DataFrame) -> "SplineAnomalyForecaster":
        """Fit the profile and ``phi`` from training data only.

        Args:
            train: Rows with ``floorspaceid``, ``timestamp``, ``target_timestamp``,
                ``lag_0m`` (current headcount) and ``target``.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``train`` is empty or columns are missing.
        """
        self._check(train, {"target"})
        if train.empty:
            raise ValueError("cannot fit on an empty training set")
        self.profile.fit(train)
        now = self.profile.evaluate(train["floorspaceid"], train["timestamp"])
        later = self.profile.evaluate(train["floorspaceid"], train["target_timestamp"])
        a = train["lag_0m"].to_numpy(dtype=float) - now
        b = train["target"].to_numpy(dtype=float) - later
        denom = float(np.sum(a * a))
        self.phi_ = float(np.sum(a * b) / denom) if denom > 0 else 0.0
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict the headcount at each row's target time.

        Args:
            df: Rows with ``floorspaceid``, ``timestamp``, ``target_timestamp`` and ``lag_0m``.

        Returns:
            Predicted headcount (people), never negative.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If columns are missing.
        """
        if self.phi_ is None:
            raise RuntimeError("SplineAnomalyForecaster must be fitted before predict")
        self._check(df, set())
        now = self.profile.evaluate(df["floorspaceid"], df["timestamp"])
        later = self.profile.evaluate(df["floorspaceid"], df["target_timestamp"])
        pred = later + self.phi_ * (df["lag_0m"].to_numpy(dtype=float) - now)
        return np.clip(pred, 0.0, None)
