"""Tests for occupancy_forecast.data_loading."""

import pandas as pd
import pytest

from occupancy_forecast.data_loading import resample_occupancy

HEADER = (
    '"id","deviceid","floorspaceid","occupancystatus","headcount",'
    '"collecteddate","occupancystatuschangedate","previousoccupancystatus"\n'
)


def _write_csv(path, rows):
    """rows: list of (floorspaceid, headcount, 'YYYY-MM-DD HH:MM:SS+TZ')."""
    lines = [HEADER]
    for i, (room, hc, ts) in enumerate(rows):
        lines.append(f'{i},"dev","{room}","CurrentlyOccupied",{hc},"{ts}","{ts}","NotOccupied"\n')
    path.write_text("".join(lines))
    return path


def test_bins_average_headcount(tmp_path):
    f = _write_csv(tmp_path / "occ.csv", [
        ("A", 2, "2024-03-01 09:00:02+11"),
        ("A", 4, "2024-03-01 09:03:00+11"),   # same 5-min bin -> mean 3
        ("A", 10, "2024-03-01 09:05:01+11"),  # next bin
    ])
    out = resample_occupancy(f, bin_minutes=5)
    assert list(out["mean_headcount"]) == [3.0, 10.0]
    assert list(out["n_obs"]) == [2, 1]


def test_timestamps_are_melbourne_local_not_utc(tmp_path):
    # 09:00 at +11 is 22:00 UTC the previous day; local hour must be 9.
    f = _write_csv(tmp_path / "occ.csv", [("A", 1, "2024-03-01 09:00:02+11")])
    out = resample_occupancy(f, bin_minutes=5)
    assert out["timestamp"].iloc[0] == pd.Timestamp("2024-03-01 09:00:00")


def test_rooms_are_kept_separate_and_sorted(tmp_path):
    f = _write_csv(tmp_path / "occ.csv", [
        ("B", 1, "2024-03-01 09:00:00+11"),
        ("A", 5, "2024-03-01 09:00:00+11"),
    ])
    out = resample_occupancy(f)
    assert list(out["floorspaceid"]) == ["A", "B"]


@pytest.mark.parametrize("bad", [0, -5, 2.5, "5", True])
def test_invalid_bin_minutes_rejected(tmp_path, bad):
    f = _write_csv(tmp_path / "occ.csv", [("A", 1, "2024-03-01 09:00:00+11")])
    with pytest.raises(ValueError):
        resample_occupancy(f, bin_minutes=bad)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resample_occupancy(tmp_path / "nope.csv")


def test_missing_required_column_raises(tmp_path):
    f = tmp_path / "bad.csv"
    f.write_text('"id","headcount"\n1,2\n')
    with pytest.raises(ValueError, match="missing required columns"):
        resample_occupancy(f)
