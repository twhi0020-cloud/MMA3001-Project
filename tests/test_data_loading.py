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


# ---------- to_local_time / bin_readings ----------

import pandas as pd  # noqa: E402

from occupancy_forecast.data_loading import bin_readings, to_local_time  # noqa: E402

_ROWS = [
    ("A", 2, "2024-03-01 09:00:02+11"), ("A", 4, "2024-03-01 09:03:00+11"), ("A", 10, "2024-03-01 09:05:01+11"),
    ("B", 1, "2024-03-01 09:00:00+11"), ("B", 5, "2024-07-02 18:11:42+10"),
]


def _readings():
    return pd.DataFrame(_ROWS, columns=["floorspaceid", "headcount", "collecteddate"])


def test_bin_readings_matches_the_duckdb_loader(tmp_path):
    f = _write_csv(tmp_path / "occ.csv", _ROWS)
    pd.testing.assert_frame_equal(bin_readings(_readings(), 5), resample_occupancy(f, 5), check_dtype=False)
    pd.testing.assert_frame_equal(bin_readings(_readings(), 15), resample_occupancy(f, 15), check_dtype=False)


def test_to_local_time_converts_offsets_to_melbourne_wall_clock():
    out = to_local_time(pd.Series(["2024-03-01 09:00:02+11", "2024-07-02 18:11:42+10", "2024-03-01T00:00:00Z"]))
    assert out.tolist() == [pd.Timestamp("2024-03-01 09:00:02"), pd.Timestamp("2024-07-02 18:11:42"),
                            pd.Timestamp("2024-03-01 11:00:00")]


def test_to_local_time_accepts_aware_datetimes_and_rejects_naive():
    aware = pd.Series(pd.to_datetime(["2024-03-01 09:00:00+11:00"]))
    assert to_local_time(aware).iloc[0] == pd.Timestamp("2024-03-01 09:00:00")
    with pytest.raises(ValueError, match="UTC offset"):
        to_local_time(pd.Series(pd.to_datetime(["2024-03-01 09:00:00"])))
    with pytest.raises(ValueError, match="UTC offset"):
        to_local_time(pd.Series(["2024-03-01 09:00:00"]))
    with pytest.raises(ValueError, match="parse"):
        to_local_time(pd.Series(["not a date+11"]))


def test_bin_readings_validation_and_empty_input():
    with pytest.raises(ValueError, match="missing required columns"):
        bin_readings(pd.DataFrame({"floorspaceid": ["A"]}))
    for bad in (0, -5, 2.5, True):
        with pytest.raises(ValueError):
            bin_readings(_readings(), bin_minutes=bad)
    assert bin_readings(_readings().iloc[0:0]).empty
