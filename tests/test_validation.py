"""Tests for occupancy_forecast.validation."""

import numpy as np
import pandas as pd
import pytest

from occupancy_forecast.validation import (
    chronological_split,
    compare_predictions,
    regression_metrics,
)

T0 = pd.Timestamp("2024-03-04 00:00:00")


def _dataset(rooms=("A", "B"), n=1000, horizon_bins=6):
    parts = []
    for room in rooms:
        ts = pd.date_range(T0, periods=n, freq="5min")
        parts.append(pd.DataFrame({
            "floorspaceid": room,
            "timestamp": ts,
            "target_timestamp": ts + pd.Timedelta(minutes=5 * horizon_bins),
            "target": np.arange(n, dtype=float),
        }))
    return pd.concat(parts, ignore_index=True)


def test_split_is_chronological_with_embargo_per_room():
    ds = _dataset()
    train, val, test = chronological_split(ds)
    for room in ("A", "B"):
        tr, va, te = (x[x["floorspaceid"] == room] for x in (train, val, test))
        assert len(tr) and len(va) and len(te)
        assert tr["target_timestamp"].max() < va["timestamp"].min()   # no target inside validation
        assert va["target_timestamp"].max() < te["timestamp"].min()   # no target inside test
        assert tr["timestamp"].max() < va["timestamp"].min() < va["timestamp"].max() < te["timestamp"].min()


def test_split_has_no_overlap_and_roughly_expected_sizes():
    ds = _dataset(rooms=("A",), n=1000)
    train, val, test = chronological_split(ds)
    ids = [set(x["timestamp"]) for x in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2]) and not (ids[0] & ids[2])
    assert abs(len(train) - 600) <= 7 and abs(len(val) - 200) <= 14 and abs(len(test) - 200) <= 2


def test_split_handles_rooms_with_different_date_ranges():
    a = _dataset(rooms=("A",), n=1000)
    b = _dataset(rooms=("B",), n=500)
    b[["timestamp", "target_timestamp"]] += pd.Timedelta(days=400)
    train, val, test = chronological_split(pd.concat([a, b], ignore_index=True))
    for part in (train, val, test):
        assert set(part["floorspaceid"]) == {"A", "B"}


@pytest.mark.parametrize("bad", [(0.5, 0.5), (0.6, 0.3, 0.3), (0.8, 0.2, 0.0), (-0.1, 0.6, 0.5)])
def test_split_rejects_bad_fractions(bad):
    with pytest.raises(ValueError):
        chronological_split(_dataset(rooms=("A",), n=50), fractions=bad)


def test_split_requires_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        chronological_split(pd.DataFrame({"floorspaceid": ["A"]}))


def test_regression_metrics_known_values():
    m = regression_metrics([1, 2, 3], [2, 2, 5])
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert m["n"] == 3


def test_regression_metrics_validation():
    with pytest.raises(ValueError, match="shape"):
        regression_metrics([1, 2], [1])
    with pytest.raises(ValueError, match="empty"):
        regression_metrics([], [])
    with pytest.raises(ValueError, match="NaN"):
        regression_metrics([1, np.nan], [1, 2])


def test_compare_predictions_uses_common_rows_only():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    table = compare_predictions(y, {"a": [1, 2, 3, 4], "b": [np.nan, 2, 3, 6]})
    assert table.loc["a", "n"] == 3 and table.loc["b", "n"] == 3     # row 0 excluded for both
    assert table.loc["a", "mae"] == 0.0
    assert table.loc["b", "mae"] == pytest.approx(2 / 3)


def test_compare_predictions_errors():
    with pytest.raises(ValueError):
        compare_predictions([1, 2], {})
    with pytest.raises(ValueError, match="shape"):
        compare_predictions([1, 2], {"a": [1]})
    with pytest.raises(ValueError, match="no rows"):
        compare_predictions([1, 2], {"a": [np.nan, np.nan]})
