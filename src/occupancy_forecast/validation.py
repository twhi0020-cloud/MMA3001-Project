"""Chronological train/validation/test splitting and error metrics.

Forecasting data must be split by time, never at random: a random split lets the model
train on moments immediately before and after each validation moment, which leaks
information and flatters the result.

Each room is split on its own timeline (60 / 20 / 20 % by default) so every room appears
in every split, even though the rooms cover different date ranges. Training rows are
removed if their *target* time reaches into the validation period (an "embargo" of one
forecast horizon), and validation rows likewise for the test period. Features of a
validation or test row may look back into earlier periods; that is normal, because that
history is available when the forecast is made.

Limitation: the rooms' splits are separate, so one room's test period can overlap another
room's training period in calendar time. Occupancy in different rooms of the same building
is correlated (e.g. holidays), so this may give slightly optimistic estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_SPLIT_COLUMNS = {"floorspaceid", "timestamp", "target_timestamp"}


def chronological_split(
    dataset: pd.DataFrame,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a supervised dataset into train, validation and test sets by time.

    Args:
        dataset: Output of ``build_supervised_dataset`` (needs ``floorspaceid``,
            ``timestamp`` and ``target_timestamp``).
        fractions: Share of each room's rows (by time order) for train, validation and
            test. Must be three positive numbers summing to 1.

    Returns:
        ``(train, validation, test)`` DataFrames. Per room, all train targets precede the
        validation period, and all validation targets precede the test period.

    Raises:
        ValueError: If ``fractions`` is invalid or required columns are missing.
    """
    if len(fractions) != 3 or any(f <= 0 for f in fractions) or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError(f"fractions must be three positive numbers summing to 1, got {fractions!r}")
    missing = _SPLIT_COLUMNS - set(dataset.columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")

    train_parts, val_parts, test_parts = [], [], []
    for _, g in dataset.groupby("floorspaceid"):
        g = g.sort_values("timestamp")
        ts = g["timestamp"].to_numpy()
        n = len(ts)
        b1 = ts[min(n - 1, int(n * fractions[0]))]
        b2 = ts[min(n - 1, int(n * (fractions[0] + fractions[1])))]
        train_parts.append(g[g["target_timestamp"] < b1])
        val_parts.append(g[(g["timestamp"] >= b1) & (g["target_timestamp"] < b2)])
        test_parts.append(g[g["timestamp"] >= b2])

    def _join(parts):
        return pd.concat(parts, ignore_index=True) if parts else dataset.iloc[0:0].copy()

    return _join(train_parts), _join(val_parts), _join(test_parts)


def regression_metrics(y_true, y_pred) -> dict:
    """Compute MAE and RMSE between true and predicted headcounts.

    Args:
        y_true: True values (people).
        y_pred: Predicted values (people), same length as ``y_true``.

    Returns:
        Dict with ``mae`` and ``rmse`` (both in people) and ``n`` (number of pairs).

    Raises:
        ValueError: If the inputs differ in length, are empty, or contain non-finite values.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("cannot compute metrics on empty arrays")
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("inputs contain NaN or infinite values")
    err = y_pred - y_true
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err**2).mean())), "n": int(err.size)}


def compare_predictions(y_true, predictions: dict) -> pd.DataFrame:
    """Score several methods on the *same* rows so the comparison is fair.

    Only rows where every method produced a finite prediction are used (a method such as
    "same time last week" is undefined early in a room's history).

    Args:
        y_true: True values (people).
        predictions: Mapping from method name to its predictions (same length as
            ``y_true``; NaN allowed where a method has no prediction).

    Returns:
        DataFrame indexed by method with columns ``mae``, ``rmse`` and ``n`` (the shared
        number of rows).

    Raises:
        ValueError: If ``predictions`` is empty, a length differs, or no common rows exist.
    """
    if not predictions:
        raise ValueError("predictions must contain at least one method")
    y_true = np.asarray(y_true, dtype=float)
    preds = {k: np.asarray(v, dtype=float) for k, v in predictions.items()}
    for name, p in preds.items():
        if p.shape != y_true.shape:
            raise ValueError(f"predictions for {name!r} have shape {p.shape}, expected {y_true.shape}")
    mask = np.isfinite(y_true)
    for p in preds.values():
        mask &= np.isfinite(p)
    if not mask.any():
        raise ValueError("no rows where all methods have a prediction")
    rows = {name: regression_metrics(y_true[mask], p[mask]) for name, p in preds.items()}
    return pd.DataFrame(rows).T[["mae", "rmse", "n"]].astype({"n": int})
