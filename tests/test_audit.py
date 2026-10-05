"""Tests for occupancy_forecast.audit."""

import json

import pandas as pd
import pytest

from occupancy_forecast.audit import (
    audit_id_match_rate,
    audit_timestamp_range,
    audit_variable_completeness,
    count_distinct,
    parse_env_sensor_csv,
)


# ---------- audit_id_match_rate ----------

def test_id_match_rate_partial_overlap():
    data_ids = {"a", "b", "c", "d"}
    lookup_ids = pd.Series(["a", "b", "x", "y"])
    result = audit_id_match_rate(data_ids, lookup_ids)
    assert result["n_data_ids"] == 4
    assert result["n_matched"] == 2
    assert result["match_rate"] == 0.5
    assert result["matched_ids"] == {"a", "b"}
    assert result["unmatched_ids"] == {"c", "d"}


def test_id_match_rate_handles_duplicates_and_na_in_lookup():
    data_ids = {"a", "b"}
    lookup_ids = pd.Series(["a", "a", None, "a"])
    result = audit_id_match_rate(data_ids, lookup_ids)
    assert result["n_matched"] == 1
    assert result["match_rate"] == 0.5


def test_id_match_rate_empty_data_ids_returns_zero_not_error():
    result = audit_id_match_rate(set(), pd.Series(["a", "b"]))
    assert result["n_data_ids"] == 0
    assert result["match_rate"] == 0.0


def test_id_match_rate_rejects_non_set():
    with pytest.raises(TypeError):
        audit_id_match_rate(["a", "b"], pd.Series(["a"]))


# ---------- count_distinct / audit_timestamp_range ----------

def _write_csv(path, header, rows):
    lines = [header + "\n"] + [r + "\n" for r in rows]
    path.write_text("".join(lines))
    return path


def test_count_distinct(tmp_path):
    f = _write_csv(
        tmp_path / "x.csv", "id,room",
        ["1,A", "2,A", "3,B", "4,B", "5,B"],
    )
    assert count_distinct(f, "room") == 2


def test_count_distinct_missing_column_raises(tmp_path):
    f = _write_csv(tmp_path / "x.csv", "id,room", ["1,A"])
    with pytest.raises(ValueError, match="not found"):
        count_distinct(f, "nope")


def test_count_distinct_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        count_distinct(tmp_path / "nope.csv", "room")


def test_timestamp_range(tmp_path):
    f = _write_csv(
        tmp_path / "x.csv", "id,ts",
        ['1,"2024-01-01 00:00:00+11"', '2,"2024-06-15 12:00:00+11"', '3,"2024-03-01 00:00:00+11"'],
    )
    result = audit_timestamp_range(f, "ts")
    assert result["min_timestamp"] == pd.Timestamp("2024-01-01 00:00:00+11:00")
    assert result["max_timestamp"] == pd.Timestamp("2024-06-15 12:00:00+11:00")
    assert result["span_days"] == pytest.approx(166.5, abs=0.01)


# ---------- parse_env_sensor_csv / audit_variable_completeness ----------

def _write_env_csv(path):
    rows = []
    for i, (sensor, ts) in enumerate([("S1", "2024-01-01T00:00:00Z"), ("S1", "2024-01-01T00:10:00Z")]):
        payload = json.dumps([
            {"variable": {"name": "Temperature", "unit": "C"}, "value": 21.5 + i},
            {"variable": {"name": "Carbon dioxide", "unit": "ppm"}, "value": 600 + i},
        ])
        rows.append(f'{i},{sensor},{ts},"{payload.replace(chr(34), chr(34)*2)}"')
    path.write_text("id,sensorid,createdate,jsondata\n" + "\n".join(rows) + "\n")
    return path


def test_parse_env_sensor_csv_unpacks_json(tmp_path):
    f = _write_env_csv(tmp_path / "env.csv")
    tidy = parse_env_sensor_csv(f)
    assert len(tidy) == 4  # 2 readings x 2 variables
    assert set(tidy["variable"]) == {"Temperature", "Carbon dioxide"}
    assert tidy["sensorid"].unique().tolist() == ["S1"]


def test_parse_env_sensor_csv_skips_bad_json(tmp_path):
    path = tmp_path / "env.csv"
    path.write_text('id,sensorid,createdate,jsondata\n0,S1,2024-01-01T00:00:00Z,"not json"\n')
    tidy = parse_env_sensor_csv(path)
    assert len(tidy) == 0


def test_variable_completeness(tmp_path):
    f = _write_env_csv(tmp_path / "env.csv")
    tidy = parse_env_sensor_csv(f)
    completeness = audit_variable_completeness(tidy)
    assert completeness.loc["Temperature", "n_present"] == 2
    assert completeness.loc["Temperature", "completeness"] == 1.0


def test_variable_completeness_missing_column_raises():
    with pytest.raises(ValueError, match="missing required columns"):
        audit_variable_completeness(pd.DataFrame({"sensorid": ["S1"]}))
