"""Load and resample the raw occupancy sensor export.

The raw file has ~9M rows sampled every 2-4 seconds. Loading it whole into pandas
is slow and memory hungry, so aggregation is pushed into DuckDB, which streams the
CSV and returns only the resampled result.

Key design decisions (see README, "Known data issues"):

* Timestamps in the export carry a UTC offset (+10/+11). They are converted to
  Australia/Melbourne *local* time before binning, otherwise hour-of-day patterns
  are shifted by 10-11 hours.
* Bins with no readings are simply absent from the output (not zero-filled);
  gap handling is a separate, explicit step.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

REQUIRED_COLUMNS = {"floorspaceid", "headcount", "collecteddate"}
LOCAL_TZ = "Australia/Melbourne"
_OFFSET = r"(?:Z|[+-]\d{2}(?::?\d{2})?)$"


def to_local_time(timestamps: pd.Series, tz: str = LOCAL_TZ) -> pd.Series:
    """Convert timestamps that carry a UTC offset to naive local wall-clock time.

    Args:
        timestamps: Strings such as ``"2024-03-01 09:00:02+11"`` (an offset is required) or
            timezone-aware datetimes.
        tz: IANA timezone to convert to.

    Returns:
        Naive local datetimes (no timezone attached).

    Raises:
        ValueError: If a value has no UTC offset (it would be ambiguous), or cannot be parsed.
    """
    if pd.api.types.is_datetime64_any_dtype(timestamps):
        if timestamps.dt.tz is None:
            raise ValueError("timestamps must include a UTC offset (got timezone-naive datetimes)")
        return timestamps.dt.tz_convert(tz).dt.tz_localize(None)
    as_text = timestamps.astype(str)
    if not as_text.str.contains(_OFFSET, regex=True).all():
        raise ValueError("timestamps must include a UTC offset, e.g. '2024-03-01 09:00:02+11'")
    try:
        parsed = pd.to_datetime(as_text, utc=True, format="ISO8601")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"could not parse timestamps: {exc}") from exc
    return parsed.dt.tz_convert(tz).dt.tz_localize(None)


def bin_readings(readings: pd.DataFrame, bin_minutes: int = 5, tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Aggregate in-memory sensor readings into fixed-width local-time bins.

    The in-memory counterpart of :func:`resample_occupancy` (same output), for the small
    amounts of recent data used when forecasting; ``resample_occupancy`` is for the large file.

    Args:
        readings: Rows with ``floorspaceid``, ``headcount`` (people) and ``collecteddate``
            (timestamp with UTC offset).
        bin_minutes: Bin width in minutes (positive integer).
        tz: IANA timezone used for local time.

    Returns:
        DataFrame sorted by (floorspaceid, timestamp) with ``floorspaceid``, ``timestamp``
        (bin start, naive local), ``mean_headcount`` and ``n_obs``.

    Raises:
        ValueError: If ``bin_minutes`` is invalid, columns are missing, or timestamps are
            invalid (see :func:`to_local_time`).
    """
    if not isinstance(bin_minutes, int) or isinstance(bin_minutes, bool) or bin_minutes <= 0:
        raise ValueError(f"bin_minutes must be a positive integer, got {bin_minutes!r}")
    missing = REQUIRED_COLUMNS - set(readings.columns)
    if missing:
        raise ValueError(f"readings is missing required columns: {sorted(missing)}")
    columns = ["floorspaceid", "timestamp", "mean_headcount", "n_obs"]
    if readings.empty:
        return pd.DataFrame(columns=columns)

    local = to_local_time(readings["collecteddate"], tz)
    grouped = (
        readings.assign(timestamp=local.dt.floor(f"{bin_minutes}min"))
        .groupby(["floorspaceid", "timestamp"])["headcount"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_headcount", "count": "n_obs"})
    )
    grouped["floorspaceid"] = grouped["floorspaceid"].astype(str)
    grouped["n_obs"] = grouped["n_obs"].astype("int64")
    return grouped[columns].sort_values(["floorspaceid", "timestamp"]).reset_index(drop=True)


def resample_occupancy(
    csv_path: str | Path,
    bin_minutes: int = 5,
    tz: str = LOCAL_TZ,
) -> pd.DataFrame:
    """Aggregate raw occupancy readings into fixed-width local-time bins.

    Args:
        csv_path: Path to the raw occupancy CSV export.
        bin_minutes: Bin width in minutes. Must be a positive integer.
        tz: IANA timezone used to convert the UTC-offset timestamps to local time.

    Returns:
        DataFrame sorted by (floorspaceid, timestamp) with columns:

        * ``floorspaceid`` (str): room / floor-space identifier.
        * ``timestamp`` (datetime64): bin start, naive local time.
        * ``mean_headcount`` (float): mean of raw ``headcount`` in the bin (people).
        * ``n_obs`` (int): number of raw readings in the bin.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If ``bin_minutes`` is not a positive integer, or the CSV is
            missing a required column.

    Note:
        Naive local timestamps are ambiguous for one hour when daylight saving
        ends. Readings in that hour are merged into the same bins.
    """
    if not isinstance(bin_minutes, int) or isinstance(bin_minutes, bool) or bin_minutes <= 0:
        raise ValueError(f"bin_minutes must be a positive integer, got {bin_minutes!r}")

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Occupancy CSV not found: {path}")

    con = duckdb.connect()
    try:
        source = "read_csv_auto(?)"
        cols = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {source}", [str(path)]).fetchall()}
        missing = REQUIRED_COLUMNS - cols
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        # tz is a parameter-safe literal only if it is a known zone; validate via DuckDB.
        con.execute("SELECT now() AT TIME ZONE ?", [tz])

        query = f"""
            SELECT
                floorspaceid,
                time_bucket(INTERVAL {bin_minutes} MINUTE,
                            collecteddate AT TIME ZONE ?) AS timestamp,
                AVG(headcount) AS mean_headcount,
                COUNT(*) AS n_obs
            FROM {source}
            GROUP BY floorspaceid, timestamp
            ORDER BY floorspaceid, timestamp
        """
        df = con.execute(query, [tz, str(path)]).fetchdf()
    finally:
        con.close()

    df["floorspaceid"] = df["floorspaceid"].astype(str)
    df["n_obs"] = df["n_obs"].astype("int64")
    return df
