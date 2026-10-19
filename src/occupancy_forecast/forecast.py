"""Forecast room headcount from raw occupancy sensor readings.

This is the project's top-level callable: :func:`forecast_headcount` takes recent readings in
the same format as the sensor export and returns, for each room, the predicted headcount one
forecast horizon ahead.

**Input** (``readings``): a DataFrame with one row per sensor reading and columns

* ``floorspaceid`` - room identifier (any hashable text);
* ``headcount`` - people counted, a non-negative number;
* ``collecteddate`` - reading time **with a UTC offset**, e.g. ``"2024-03-01 09:00:02+11"``
  (naive times are rejected because they are ambiguous around daylight saving).

At least about 65 minutes of history is needed before a forecast is possible; fewer rows give
``status = "insufficient_history"`` and a missing prediction.

**Output**: one row per room with ``floorspaceid``, ``as_of`` (start of the last complete
5-minute bin, Melbourne local time), ``forecast_for`` (start of the bin being predicted,
``as_of`` plus the horizon), ``predicted_headcount`` (people, never negative),
``current_headcount`` (the latest value used), ``status`` and ``outside_training_range``.
The forecast is issued at the *end* of the ``as_of`` bin, so the lead time is the horizon minus
one bin width.

**Handling of unusual input.** Rows with a missing or non-numeric headcount are dropped (their
number is reported in ``result.attrs["dropped_readings"]``). Negative headcounts, missing
columns, timestamps without a UTC offset, and an empty table raise ``ValueError``. Readings
after ``as_of`` are ignored. A decision tree cannot extrapolate: outside its training range it
returns the value of the nearest leaf. ``outside_training_range`` is therefore ``True`` when
any feature lies beyond the range seen in training, and such a forecast should be treated with
caution.

The model itself is a *bundle* (a dict, normally saved with :func:`save_bundle` by
``scripts/train_final.py``) holding the fitted tree, its feature names, horizon and the range
of each feature in the training data.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from occupancy_forecast.cleaning import clean_occupancy
from occupancy_forecast.data_loading import LOCAL_TZ, bin_readings
from occupancy_forecast.experiments import scaled_clean_kwargs
from occupancy_forecast.features import (
    DEFAULT_LAGS_MINUTES,
    DEFAULT_ROLLING_MINUTES,
    feature_names,
    forecast_features,
)

BUNDLE_KEYS = {"model", "features", "horizon_minutes", "bin_minutes", "lags_minutes",
               "rolling_minutes", "feature_ranges", "params"}
OUTPUT_COLUMNS = ["floorspaceid", "as_of", "forecast_for", "predicted_headcount",
                  "current_headcount", "status", "outside_training_range"]


def make_bundle(models, bin_minutes: int = 5, lags_minutes=DEFAULT_LAGS_MINUTES,
                rolling_minutes=DEFAULT_ROLLING_MINUTES) -> dict:
    """Package the fitted tree and everything needed to use it into one dict.

    Args:
        models: A ``FinalModels`` from :func:`occupancy_forecast.final.fit_final_models`.
        bin_minutes: Bin width the model was trained with.
        lags_minutes: Lags the model was trained with.
        rolling_minutes: Rolling windows the model was trained with.

    Returns:
        The bundle dict (see the module documentation).
    """
    return {
        "model": models.tree, "features": list(models.features), "horizon_minutes": models.horizon_minutes,
        "bin_minutes": bin_minutes, "lags_minutes": tuple(lags_minutes), "rolling_minutes": tuple(rolling_minutes),
        "feature_ranges": dict(models.feature_ranges), "params": dict(models.best_params),
    }


def _validate_bundle(bundle) -> dict:
    if not isinstance(bundle, dict):
        raise ValueError("model must be a bundle dict or a path to a saved bundle")
    missing = BUNDLE_KEYS - set(bundle)
    if missing:
        raise ValueError(f"model bundle is missing keys: {sorted(missing)}")
    expected = feature_names(tuple(bundle["lags_minutes"]), tuple(bundle["rolling_minutes"]))
    if list(bundle["features"]) != expected:
        raise ValueError("model bundle features do not match its lags and rolling windows")
    return bundle


def save_bundle(bundle: dict, path: str | Path) -> Path:
    """Validate and save a model bundle.

    Args:
        bundle: Bundle from :func:`make_bundle`.
        path: Destination file (parent folders are created).

    Returns:
        The path written.

    Raises:
        ValueError: If the bundle is invalid.
    """
    _validate_bundle(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str | Path) -> dict:
    """Load and validate a saved model bundle.

    Args:
        path: File written by :func:`save_bundle`.

    Returns:
        The bundle dict.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file does not contain a valid bundle.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"model bundle not found: {path}")
    return _validate_bundle(joblib.load(path))


def forecast_headcount(readings: pd.DataFrame, model, as_of=None, tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Predict each room's headcount one horizon ahead from recent raw readings.

    Args:
        readings: Raw sensor readings (columns ``floorspaceid``, ``headcount``,
            ``collecteddate``; see the module documentation).
        model: A model bundle dict, or the path to one saved with :func:`save_bundle`.
        as_of: Time of the last complete bin. A timezone-aware value is converted to local
            time; a naive value is taken as local. Defaults to the latest reading's bin.
            Readings after this bin are ignored.
        tz: IANA timezone for local time.

    Returns:
        One row per room, with the columns listed in the module documentation.
        ``result.attrs["dropped_readings"]`` holds the number of rows dropped for a missing
        headcount.

    Raises:
        ValueError: If ``readings`` is empty or lacks required columns, a headcount is
            negative, timestamps lack a UTC offset, the bundle is invalid, or there are no
            readings at or before ``as_of``.
    """
    bundle = load_bundle(model) if isinstance(model, (str, Path)) else _validate_bundle(model)
    if not isinstance(readings, pd.DataFrame):
        raise ValueError("readings must be a pandas DataFrame")
    missing = {"floorspaceid", "headcount", "collecteddate"} - set(readings.columns)
    if missing:
        raise ValueError(f"readings is missing required columns: {sorted(missing)}")
    if readings.empty:
        raise ValueError("readings is empty")

    headcount = pd.to_numeric(readings["headcount"], errors="coerce")
    if (headcount < 0).any():
        raise ValueError("readings contain negative headcounts, which are not valid")
    keep = headcount.notna() & readings["floorspaceid"].notna()
    dropped = int((~keep).sum())
    cleaned_in = readings.loc[keep].assign(headcount=headcount[keep])
    if cleaned_in.empty:
        raise ValueError("no usable readings (every headcount is missing)")

    bin_minutes = int(bundle["bin_minutes"])
    binned = bin_readings(cleaned_in, bin_minutes, tz)

    if as_of is None:
        as_of_bin = binned["timestamp"].max()
    else:
        stamp = pd.Timestamp(as_of)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert(tz).tz_localize(None)
        as_of_bin = stamp.floor(f"{bin_minutes}min")
    binned = binned[binned["timestamp"] <= as_of_bin]
    if binned.empty:
        raise ValueError("no readings at or before as_of")

    # Extend each room's grid to as_of with an empty bin, so a room that has gone quiet is
    # handled by the same gap rules as in training instead of silently using stale data.
    last = binned.groupby("floorspaceid")["timestamp"].max()
    stale = last[last < as_of_bin]
    if len(stale):
        pad = pd.DataFrame({"floorspaceid": stale.index.astype(str), "timestamp": as_of_bin,
                            "mean_headcount": np.nan, "n_obs": 0})
        binned = pd.concat([binned, pad], ignore_index=True)

    clean = clean_occupancy(binned, **scaled_clean_kwargs(bin_minutes))
    feats = forecast_features(
        clean, as_of_bin, horizon_minutes=int(bundle["horizon_minutes"]), bin_minutes=bin_minutes,
        lags_minutes=tuple(bundle["lags_minutes"]), rolling_minutes=tuple(bundle["rolling_minutes"]),
    )

    names = list(bundle["features"])
    complete = feats["complete"].to_numpy()
    prediction = np.full(len(feats), np.nan)
    if complete.any():
        prediction[complete] = np.clip(bundle["model"].predict(feats.loc[complete, names]), 0.0, None)

    outside = np.zeros(len(feats), dtype=bool)
    for f, (lo, hi) in bundle["feature_ranges"].items():
        outside |= complete & ((feats[f] < lo) | (feats[f] > hi)).to_numpy()

    result = pd.DataFrame({
        "floorspaceid": feats["floorspaceid"].to_numpy(),
        "as_of": as_of_bin,
        "forecast_for": feats["target_timestamp"].to_numpy(),
        "predicted_headcount": prediction,
        "current_headcount": feats["lag_0m"].to_numpy() if "lag_0m" in feats else np.nan,
        "status": np.where(complete, "ok", "insufficient_history"),
        "outside_training_range": outside,
    })[OUTPUT_COLUMNS]
    result.attrs["dropped_readings"] = dropped
    return result
