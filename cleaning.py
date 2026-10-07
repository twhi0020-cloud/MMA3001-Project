"""Gap and glitch handling for resampled occupancy data.

The resampled 5-minute table from :func:`occupancy_forecast.data_loading.resample_occupancy`
only contains bins that received readings. The audit showed that:

* coverage of the full 5-minute grid is only ~36-58 % per room;
* missing bins cluster overnight and at weekends, and the headcount just before a gap
  is usually low, which suggests the hub often stays quiet when a room is empty rather
  than dropping data at random;
* headcount is heavy-tailed: a few bins hold implausibly large values.

This module therefore applies three explicit, parameterised steps, each of which keeps
the original value and records what was done:

1. :func:`regularize_grid` - put every room on a complete, regular time grid.
2. :func:`cap_spikes` - flag and cap upward outliers using a rolling median / MAD rule.
3. :func:`fill_gaps` - interpolate short gaps, fill long "quiet" gaps with zero, and
   leave everything else missing.

:func:`clean_occupancy` runs all three. The ``assumed_empty`` rule is an inference from
indirect evidence (there is no ground truth for why the hub is quiet), so it is exposed
as a parameter to allow sensitivity testing.

Known limitation: timestamps are naive Melbourne local time, so the hour skipped when
daylight saving starts appears as a 12-bin gap (at ~2 am Sunday, normally when rooms
are empty).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FILL_OBSERVED = "observed"
FILL_INTERPOLATED = "interpolated"
FILL_ASSUMED_EMPTY = "assumed_empty"
FILL_MISSING = "missing"

_INPUT_COLUMNS = {"floorspaceid", "timestamp", "mean_headcount", "n_obs"}


def _check_positive_int(name: str, value) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _check_non_negative_number(name: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name} must be a non-negative number, got {value!r}")


def regularize_grid(df: pd.DataFrame, bin_minutes: int = 5) -> pd.DataFrame:
    """Reindex each room onto a complete, regular time grid.

    Args:
        df: Output of ``resample_occupancy`` with columns ``floorspaceid``,
            ``timestamp``, ``mean_headcount`` and ``n_obs``.
        bin_minutes: Bin width in minutes; must match the width used to resample.

    Returns:
        DataFrame sorted by (floorspaceid, timestamp) with columns:

        * ``floorspaceid`` (str)
        * ``timestamp`` (datetime64): every bin from each room's first to last reading.
        * ``mean_headcount_raw`` (float): observed mean headcount (people); NaN where
          the bin had no readings.
        * ``n_obs`` (int): raw readings in the bin; 0 for added bins.

    Raises:
        ValueError: If required columns are missing, ``bin_minutes`` is invalid,
            timestamps are not aligned to the bin width, or a room has duplicate
            timestamps.
    """
    _check_positive_int("bin_minutes", bin_minutes)
    missing = _INPUT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    columns = ["floorspaceid", "timestamp", "mean_headcount_raw", "n_obs"]
    if df.empty:
        return pd.DataFrame(columns=columns)

    ts = pd.to_datetime(df["timestamp"])
    if ((ts - ts.dt.floor(f"{bin_minutes}min")) != pd.Timedelta(0)).any():
        raise ValueError(f"timestamps are not aligned to {bin_minutes}-minute bins")

    freq = pd.Timedelta(minutes=bin_minutes)
    pieces = []
    for room, g in df.groupby("floorspaceid", sort=True):
        g = g.assign(timestamp=pd.to_datetime(g["timestamp"])).sort_values("timestamp")
        g = g.set_index("timestamp")
        if g.index.has_duplicates:
            raise ValueError(f"duplicate timestamps for floorspaceid {room!r}")
        full = pd.date_range(g.index.min(), g.index.max(), freq=freq)
        g = g.reindex(full)
        g["floorspaceid"] = room
        g["n_obs"] = g["n_obs"].fillna(0).astype("int64")
        g.index.name = "timestamp"
        pieces.append(g.reset_index())

    out = pd.concat(pieces, ignore_index=True)
    out = out.rename(columns={"mean_headcount": "mean_headcount_raw"})
    return out[columns]


def cap_spikes(
    df: pd.DataFrame,
    window_bins: int = 49,
    n_sigma: float = 5.0,
    min_scale: float = 1.0,
    min_periods: int = 8,
) -> pd.DataFrame:
    """Flag and cap implausibly large headcount values using a robust rolling rule.

    For each room, a centred rolling median and median absolute deviation (MAD) of
    the observed bins define a local upper limit::

        upper = median + n_sigma * max(1.4826 * MAD, min_scale)

    Values above ``upper`` are flagged and capped to ``upper``. Median/MAD are used
    because a few extreme values barely move them, unlike a mean/standard deviation.
    ``min_scale`` stops quiet periods (MAD near zero) from flagging ordinary
    fluctuations: a value must be at least ``n_sigma * min_scale`` people above the
    local median. Only upward spikes are capped; headcount cannot be negative.

    Args:
        df: Output of :func:`regularize_grid` (regular grid per room).
        window_bins: Rolling window length in bins (49 bins = ~4 h at 5 min).
        n_sigma: Multiplier on the robust scale. Larger flags fewer values.
        min_scale: Minimum robust scale, in people.
        min_periods: Minimum observed bins in the window for a limit to be computed;
            bins with fewer are never flagged.

    Returns:
        Copy of ``df`` with added columns ``is_spike`` (bool) and
        ``headcount_capped`` (float; equals ``mean_headcount_raw`` except where capped).

    Raises:
        ValueError: If parameters are invalid, ``mean_headcount_raw`` is missing, or a
            room is not on a regular time grid.
    """
    _check_positive_int("window_bins", window_bins)
    _check_positive_int("min_periods", min_periods)
    _check_non_negative_number("n_sigma", n_sigma)
    _check_non_negative_number("min_scale", min_scale)
    if "mean_headcount_raw" not in df.columns:
        raise ValueError("df must contain 'mean_headcount_raw' (run regularize_grid first)")

    out = df.copy().reset_index(drop=True)
    out["is_spike"] = False
    out["headcount_capped"] = out["mean_headcount_raw"]

    for room, idx in out.groupby("floorspaceid").groups.items():
        steps = out.loc[idx, "timestamp"].diff().dropna().nunique()
        if steps > 1:
            raise ValueError(
                f"floorspaceid {room!r} is not on a regular time grid; run regularize_grid first"
            )
        x = out.loc[idx, "mean_headcount_raw"]
        med = x.rolling(window_bins, center=True, min_periods=min_periods).median()
        mad = (x - med).abs().rolling(window_bins, center=True, min_periods=min_periods).median()
        scale = (1.4826 * mad).clip(lower=min_scale)
        upper = med + n_sigma * scale
        spike = x > upper  # NaN comparisons are False, so missing bins are never flagged
        out.loc[idx, "is_spike"] = spike
        out.loc[idx, "headcount_capped"] = x.where(~spike, upper)
    return out


def _fill_series(
    values: np.ndarray,
    max_interp_bins: int,
    quiet_threshold: float,
    max_quiet_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill the NaN runs of one room's series; returns (filled values, status labels)."""
    n = len(values)
    filled = values.astype(float).copy()
    isnan = np.isnan(values)
    status = np.where(isnan, FILL_MISSING, FILL_OBSERVED).astype(object)

    edges = np.diff(np.concatenate(([0], isnan.astype(int), [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]  # exclusive

    for s, e in zip(starts, ends):
        length = e - s
        prev = values[s - 1] if s > 0 else np.nan
        nxt = values[e] if e < n else np.nan
        if length <= max_interp_bins and not np.isnan(prev) and not np.isnan(nxt):
            filled[s:e] = np.interp(np.arange(s, e), [s - 1, e], [prev, nxt])
            status[s:e] = FILL_INTERPOLATED
        elif length <= max_quiet_bins and not np.isnan(prev) and prev <= quiet_threshold:
            filled[s:e] = 0.0
            status[s:e] = FILL_ASSUMED_EMPTY
    return filled, status


def fill_gaps(
    df: pd.DataFrame,
    value_col: str = "headcount_capped",
    max_interp_bins: int = 3,
    quiet_threshold: float = 0.5,
    max_quiet_bins: int = 288,
) -> pd.DataFrame:
    """Fill gaps in the headcount series according to explicit rules.

    Each run of consecutive missing bins in a room is treated as follows:

    1. If it is at most ``max_interp_bins`` long and has observed values on both
       sides, it is linearly interpolated (``interpolated``).
    2. Otherwise, if it is at most ``max_quiet_bins`` long and the last observed value
       before it is at most ``quiet_threshold``, it is filled with 0 people
       (``assumed_empty``). This encodes the inference that the hub stays quiet when a
       room is empty; it is not verified against ground truth.
    3. Otherwise it stays NaN (``missing``) and should be excluded from training and
       evaluation.

    Args:
        df: DataFrame on a regular grid (see :func:`regularize_grid`).
        value_col: Column to fill.
        max_interp_bins: Longest gap (bins) to interpolate. 3 bins = 15 min at 5 min.
        quiet_threshold: Headcount (people) at or below which a preceding value counts
            as "room empty".
        max_quiet_bins: Longest gap (bins) to fill with zero. 288 bins = 1 day at 5 min.

    Returns:
        Copy of ``df`` with added columns ``headcount`` (float, people) and
        ``fill_status`` (one of ``observed``, ``interpolated``, ``assumed_empty``,
        ``missing``).

    Raises:
        ValueError: If parameters are invalid or ``value_col`` is missing.
    """
    _check_positive_int("max_interp_bins", max_interp_bins)
    _check_positive_int("max_quiet_bins", max_quiet_bins)
    _check_non_negative_number("quiet_threshold", quiet_threshold)
    if value_col not in df.columns:
        raise ValueError(f"column {value_col!r} not found in df")

    out = df.copy().reset_index(drop=True)
    out["headcount"] = np.nan
    out["fill_status"] = FILL_MISSING
    for _, idx in out.groupby("floorspaceid").groups.items():
        values = out.loc[idx, value_col].to_numpy(dtype=float)
        filled, status = _fill_series(values, max_interp_bins, quiet_threshold, max_quiet_bins)
        out.loc[idx, "headcount"] = filled
        out.loc[idx, "fill_status"] = status
    return out


def clean_occupancy(
    df: pd.DataFrame,
    bin_minutes: int = 5,
    window_bins: int = 49,
    n_sigma: float = 5.0,
    min_scale: float = 1.0,
    min_periods: int = 8,
    max_interp_bins: int = 3,
    quiet_threshold: float = 0.5,
    max_quiet_bins: int = 288,
) -> pd.DataFrame:
    """Run grid regularisation, spike capping and gap filling in sequence.

    Spikes are capped on observed values *before* gaps are filled, so interpolation
    never bridges to an outlier. See the individual functions for parameter meaning.

    Args:
        df: Output of ``resample_occupancy``.
        bin_minutes: Bin width in minutes used when resampling.
        window_bins: See :func:`cap_spikes`.
        n_sigma: See :func:`cap_spikes`.
        min_scale: See :func:`cap_spikes`.
        min_periods: See :func:`cap_spikes`.
        max_interp_bins: See :func:`fill_gaps`.
        quiet_threshold: See :func:`fill_gaps`.
        max_quiet_bins: See :func:`fill_gaps`.

    Returns:
        DataFrame with columns ``floorspaceid``, ``timestamp``, ``mean_headcount_raw``,
        ``n_obs``, ``is_spike``, ``headcount`` (cleaned, people) and ``fill_status``.
    """
    gridded = regularize_grid(df, bin_minutes=bin_minutes)
    capped = cap_spikes(
        gridded, window_bins=window_bins, n_sigma=n_sigma,
        min_scale=min_scale, min_periods=min_periods,
    )
    filled = fill_gaps(
        capped, max_interp_bins=max_interp_bins,
        quiet_threshold=quiet_threshold, max_quiet_bins=max_quiet_bins,
    )
    return filled[[
        "floorspaceid", "timestamp", "mean_headcount_raw", "n_obs",
        "is_spike", "headcount", "fill_status",
    ]]


def cleaning_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise what cleaning did to each room, for reporting.

    Args:
        df: Output of :func:`clean_occupancy`.

    Returns:
        DataFrame indexed by ``floorspaceid`` with the number of grid bins
        (``n_bins``), the share of bins in each ``fill_status`` (columns named after
        the statuses), the number of capped spikes (``n_spikes``) and the largest raw
        headcount (``max_raw``).

    Raises:
        ValueError: If a required column is missing.
    """
    required = {"floorspaceid", "fill_status", "is_spike", "mean_headcount_raw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    shares = (
        df.groupby("floorspaceid")["fill_status"].value_counts(normalize=True).unstack(fill_value=0.0)
    )
    for status in (FILL_OBSERVED, FILL_INTERPOLATED, FILL_ASSUMED_EMPTY, FILL_MISSING):
        if status not in shares.columns:
            shares[status] = 0.0
    shares = shares[[FILL_OBSERVED, FILL_INTERPOLATED, FILL_ASSUMED_EMPTY, FILL_MISSING]]

    g = df.groupby("floorspaceid")
    shares.insert(0, "n_bins", g.size())
    shares["n_spikes"] = g["is_spike"].sum().astype(int)
    shares["max_raw"] = g["mean_headcount_raw"].max()
    return shares
