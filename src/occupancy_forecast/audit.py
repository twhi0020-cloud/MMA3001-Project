"""Reproducible data-quality audit checks for the occupancy and environmental datasets.

These functions turn the exploratory findings from the project's data-inventory phase
into reusable, tested checks:

* how well sensor/room IDs in the raw data match the supplied location lookup
  (``Sensor_ID_and_Locations.xlsx``, whose OCC sheet is flagged "Incorrect Data" by
  its authors);
* whether the timestamp range in a CSV matches expectations (the occupancy and
  environmental files are named "MayToDec2024" but contain a wider range);
* how many distinct values a column takes (e.g. confirming the occupancy data comes
  from a single hub device covering several rooms);
* how completely the environmental sensors' nested-JSON readings populate each
  variable (CO2, temperature, etc.).

Each function returns plain Python / pandas objects so results can be asserted on in
tests and reported directly in the write-up.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd


def audit_id_match_rate(data_ids: set[str], lookup_ids: pd.Series) -> dict:
    """Compare a set of IDs seen in sensor data against a location lookup table.

    Args:
        data_ids: IDs observed in the raw sensor data (e.g. ``sensorid`` or
            ``floorspaceid`` values).
        lookup_ids: The corresponding ID column from the location lookup sheet.
            Not required to be unique; duplicates are ignored.

    Returns:
        A dict with:

        * ``n_data_ids`` (int): number of distinct IDs in ``data_ids``.
        * ``n_matched`` (int): how many of those IDs appear in ``lookup_ids``.
        * ``match_rate`` (float): ``n_matched / n_data_ids``, or 0.0 if
          ``data_ids`` is empty.
        * ``matched_ids`` (set[str]): the IDs that were found.
        * ``unmatched_ids`` (set[str]): the IDs that were not found.

    Raises:
        TypeError: If ``data_ids`` is not a set.
    """
    if not isinstance(data_ids, set):
        raise TypeError(f"data_ids must be a set, got {type(data_ids).__name__}")

    lookup_set = set(lookup_ids.dropna().astype(str))
    data_ids = {str(i) for i in data_ids}

    matched = data_ids & lookup_set
    unmatched = data_ids - lookup_set
    n_data_ids = len(data_ids)

    return {
        "n_data_ids": n_data_ids,
        "n_matched": len(matched),
        "match_rate": (len(matched) / n_data_ids) if n_data_ids else 0.0,
        "matched_ids": matched,
        "unmatched_ids": unmatched,
    }


def count_distinct(csv_path: str | Path, column: str) -> int:
    """Count distinct values of one column in a (possibly large) CSV file.

    Uses DuckDB so the file is streamed rather than loaded fully into memory;
    safe to call on multi-gigabyte exports.

    Args:
        csv_path: Path to the CSV file.
        column: Name of the column to count distinct values of.

    Returns:
        The number of distinct non-null values in ``column``.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If ``column`` is not present in the CSV.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    con = duckdb.connect()
    try:
        cols = {row[0] for row in con.execute(
            "DESCRIBE SELECT * FROM read_csv_auto(?)", [str(path)]
        ).fetchall()}
        if column not in cols:
            raise ValueError(f"Column {column!r} not found in {path.name}. Available: {sorted(cols)}")

        result = con.execute(
            f"SELECT COUNT(DISTINCT {column}) FROM read_csv_auto(?)", [str(path)]
        ).fetchone()
        return int(result[0])
    finally:
        con.close()


def audit_timestamp_range(csv_path: str | Path, timestamp_col: str) -> dict:
    """Report the min/max timestamp in a CSV, to check against an expected range.

    Useful for checking a file's name/description (e.g. "MayToDec2024") against
    what the data actually contains.

    Args:
        csv_path: Path to the CSV file.
        timestamp_col: Name of the timestamp column.

    Returns:
        A dict with ``min_timestamp``, ``max_timestamp`` (both ``pandas.Timestamp``),
        and ``span_days`` (float).

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If ``timestamp_col`` is not present in the CSV.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    con = duckdb.connect()
    try:
        cols = {row[0] for row in con.execute(
            "DESCRIBE SELECT * FROM read_csv_auto(?)", [str(path)]
        ).fetchall()}
        if timestamp_col not in cols:
            raise ValueError(
                f"Column {timestamp_col!r} not found in {path.name}. Available: {sorted(cols)}"
            )

        row = con.execute(
            f"SELECT MIN({timestamp_col}), MAX({timestamp_col}) FROM read_csv_auto(?)",
            [str(path)],
        ).fetchone()
    finally:
        con.close()

    min_ts, max_ts = pd.Timestamp(row[0]), pd.Timestamp(row[1])
    return {
        "min_timestamp": min_ts,
        "max_timestamp": max_ts,
        "span_days": (max_ts - min_ts).total_seconds() / 86400,
    }


def parse_env_sensor_csv(csv_path: str | Path) -> pd.DataFrame:
    """Parse the environmental sensor export's nested-JSON readings into tidy form.

    Each raw row holds one ``jsondata`` payload containing several variables
    (Battery, Carbon dioxide, Humidity, etc.) for one sensor at one time. This
    unpacks that into one row per (sensor, timestamp, variable) observation.

    Args:
        csv_path: Path to the raw environmental sensor CSV
            (expects ``sensorid``, ``createdate``, ``jsondata`` columns).

    Returns:
        Tidy DataFrame with columns ``sensorid``, ``timestamp``, ``variable``,
        ``value``, ``unit``.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    raw = pd.read_csv(path)
    records = []
    for sensorid, createdate, jsondata in zip(raw["sensorid"], raw["createdate"], raw["jsondata"]):
        try:
            entries = json.loads(jsondata)
        except (TypeError, ValueError):
            continue
        for entry in entries:
            records.append({
                "sensorid": str(sensorid),
                "timestamp": createdate,
                "variable": entry.get("variable", {}).get("name"),
                "value": entry.get("value"),
                "unit": entry.get("variable", {}).get("unit"),
            })

    tidy = pd.DataFrame.from_records(
        records, columns=["sensorid", "timestamp", "variable", "value", "unit"]
    )
    tidy["timestamp"] = pd.to_datetime(tidy["timestamp"], utc=True, errors="coerce")
    return tidy


def audit_variable_completeness(tidy_env_df: pd.DataFrame) -> pd.DataFrame:
    """Report how often each environmental variable appears per sensor reading.

    Args:
        tidy_env_df: Output of :func:`parse_env_sensor_csv`.

    Returns:
        DataFrame with one row per ``variable``, columns ``n_present`` and
        ``completeness`` (``n_present`` divided by the number of distinct
        (sensorid, timestamp) readings).

    Raises:
        ValueError: If ``tidy_env_df`` is missing a required column.
    """
    required = {"sensorid", "timestamp", "variable"}
    missing = required - set(tidy_env_df.columns)
    if missing:
        raise ValueError(f"tidy_env_df is missing required columns: {sorted(missing)}")

    n_readings = tidy_env_df.drop_duplicates(["sensorid", "timestamp"]).shape[0]
    counts = tidy_env_df.groupby("variable").size().rename("n_present").to_frame()
    counts["completeness"] = counts["n_present"] / n_readings if n_readings else 0.0
    return counts.sort_values("n_present", ascending=False)
