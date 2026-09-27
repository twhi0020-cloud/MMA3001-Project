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
