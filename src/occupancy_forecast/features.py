"""Build the supervised learning table for forecasting headcount ``h`` minutes ahead.

Each row is one forecast made at time ``t`` for one room. Every feature uses only
information available at or before ``t``; the target is the headcount at ``t + h``.

Features (``FEATURE_COLUMNS`` for the default settings):

* ``lag_{m}m`` - headcount ``m`` minutes before ``t`` (``lag_0m`` is the value at ``t``).
* ``roll_mean_{m}m`` - mean headcount over the ``m`` minutes ending at ``t``.
* ``hour_of_day`` - hour of the **target** time, as a fraction (09:30 -> 9.5).
* ``day_of_week`` - day of the **target** time (Monday = 0).

Calendar features describe the target time rather than ``t`` because the expected
pattern depends on when the forecast applies; for a fixed horizon the two are
equivalent information, so only one is included.

The room identity is deliberately *not* a feature, so the model is not tied to the five
rooms in the data; the lag and rolling features carry each room's typical level.

Rows are dropped when the target or any feature is missing (``fill_status == 'missing'``
bins in the cleaned data). The ``baseline_last_week`` column (headcount at the target
time one week earlier) is kept for baseline comparison but is **not** a model feature,
so requiring it would not shrink the training set.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {"floorspaceid", "timestamp", "headcount", "fill_status"}
DEFAULT_LAGS_MINUTES = (0, 15, 30, 60)
DEFAULT_ROLLING_MINUTES = (30, 60)
CALENDAR_FEATURES = ["hour_of_day", "day_of_week"]
WEEK_MINUTES = 7 * 24 * 60


def feature_names(
    lags_minutes: tuple[int, ...] = DEFAULT_LAGS_MINUTES,
    rolling_minutes: tuple[int, ...] = DEFAULT_ROLLING_MINUTES,
) -> list[str]:
    """Return the model feature column names for the given lag/rolling settings.

    Args:
        lags_minutes: Lags in minutes (0 = current value).
        rolling_minutes: Rolling-mean window lengths in minutes.

    Returns:
        Column names in the order they appear in the dataset.
    """
    return (
        [f"lag_{m}m" for m in lags_minutes]
        + [f"roll_mean_{m}m" for m in rolling_minutes]
        + CALENDAR_FEATURES
    )


def _bins(name: str, minutes: int, bin_minutes: int, minimum: int) -> int:
    if isinstance(minutes, bool) or not isinstance(minutes, int) or minutes < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {minutes!r}")
    if minutes % bin_minutes:
        raise ValueError(f"{name} ({minutes}) must be a multiple of bin_minutes ({bin_minutes})")
    return minutes // bin_minutes


def build_supervised_dataset(
    clean_df: pd.DataFrame,
    horizon_minutes: int = 30,
    bin_minutes: int = 5,
    lags_minutes: tuple[int, ...] = DEFAULT_LAGS_MINUTES,
    rolling_minutes: tuple[int, ...] = DEFAULT_ROLLING_MINUTES,
) -> pd.DataFrame:
    """Turn cleaned occupancy data into (features, target) rows for one horizon.

    Args:
        clean_df: Output of ``clean_occupancy``; needs ``floorspaceid``, ``timestamp``,
            ``headcount`` (people, NaN where missing) and ``fill_status``. Each room must
            be on a regular grid of ``bin_minutes``.
        horizon_minutes: Forecast horizon ``h`` in minutes (positive multiple of
            ``bin_minutes``).
        bin_minutes: Grid spacing in minutes.
        lags_minutes: Past lags to use as features (multiples of ``bin_minutes``, >= 0).
        rolling_minutes: Rolling-mean windows ending at ``t`` (multiples of
            ``bin_minutes``, >= ``bin_minutes``).

    Returns:
        DataFrame sorted by (floorspaceid, timestamp) with columns:

        * ``floorspaceid``, ``timestamp`` (forecast time ``t``), ``target_timestamp``
          (``t + h``);
        * the feature columns from :func:`feature_names`;
        * ``target`` (people at ``t + h``) and ``target_fill_status`` (how that value was
          obtained: ``observed``, ``interpolated`` or ``assumed_empty``);
        * ``baseline_last_week`` (people at ``t + h`` minus 7 days; NaN if unavailable).

        Rows whose target or any feature is missing are removed.

    Raises:
        ValueError: If required columns are missing, an argument is invalid, or a room is
            not on a regular grid.
    """
    missing = REQUIRED_COLUMNS - set(clean_df.columns)
    if missing:
        raise ValueError(f"clean_df is missing required columns: {sorted(missing)}")
    if isinstance(bin_minutes, bool) or not isinstance(bin_minutes, int) or bin_minutes <= 0:
        raise ValueError(f"bin_minutes must be a positive integer, got {bin_minutes!r}")

    h = _bins("horizon_minutes", horizon_minutes, bin_minutes, bin_minutes)
    lag_bins = [(m, _bins("lag", m, bin_minutes, 0)) for m in lags_minutes]
    roll_bins = [(m, _bins("rolling window", m, bin_minutes, bin_minutes)) for m in rolling_minutes]
    week = _bins("one week", WEEK_MINUTES, bin_minutes, bin_minutes)

    df = clean_df.sort_values(["floorspaceid", "timestamp"]).reset_index(drop=True)
    step = pd.Timedelta(minutes=bin_minutes)
    for room, g in df.groupby("floorspaceid"):
        diffs = g["timestamp"].diff().dropna()
        if not (diffs == step).all():
            raise ValueError(
                f"floorspaceid {room!r} is not on a regular {bin_minutes}-minute grid; "
                "run clean_occupancy first"
            )

    by_room = df.groupby("floorspaceid")["headcount"]
    out = pd.DataFrame({"floorspaceid": df["floorspaceid"], "timestamp": df["timestamp"]})
    out["target_timestamp"] = df["timestamp"] + h * step

    for minutes, n in lag_bins:
        out[f"lag_{minutes}m"] = by_room.shift(n)
    for minutes, n in roll_bins:
        out[f"roll_mean_{minutes}m"] = by_room.transform(
            lambda s, n=n: s.rolling(n, min_periods=n).mean()
        )

    tt = out["target_timestamp"]
    out["hour_of_day"] = tt.dt.hour + tt.dt.minute / 60.0
    out["day_of_week"] = tt.dt.dayofweek

    out["target"] = by_room.shift(-h)
    out["target_fill_status"] = df.groupby("floorspaceid")["fill_status"].shift(-h)
    out["baseline_last_week"] = by_room.shift(week - h)

    needed = feature_names(lags_minutes, rolling_minutes) + ["target"]
    out = out.dropna(subset=needed).reset_index(drop=True)
    out["day_of_week"] = out["day_of_week"].astype(int)
    return out
