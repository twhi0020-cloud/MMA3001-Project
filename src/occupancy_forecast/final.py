"""Final models, a strictly-causal check, and bootstrap uncertainty for the test-set evaluation.

**Protocol** (fixed before the sealed test set was opened; see ``scripts/final_evaluation.py``):

1. Choose the tree's settings with the fast tuner using the *training* rows and scoring on the
   *validation* rows (:func:`fit_final_models`).
2. Refit every learned method (tree, linear regression, seasonal average, spline profile and
   hybrid) on training **plus** validation rows, so all methods get the same data.
3. Score every method once on the test rows, on identical rows, and report uncertainty with a
   block bootstrap over whole room-days (:func:`block_bootstrap`).

Because the test rows are never used to choose or fit anything, their error is an honest
estimate of forecasting error on later, unseen periods.

**A caveat the validation split cannot show.** The cleaning step interpolates short gaps using
the *next* observed value and caps spikes using a window that looks ahead. In training this
puts a little future information into some features; a live system cannot see the future.
:func:`lookahead_flags` marks the rows affected so results can be re-scored on rows that are
strictly causal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from occupancy_forecast.features import feature_names
from occupancy_forecast.models import (
    SeasonalAverage,
    fit_decision_tree,
    fit_linear,
    predict_last_week,
    predict_persistence,
    tune_decision_tree_fast,
)
from occupancy_forecast.splines import SplineAnomalyForecaster, SplineProfile

FINAL_METHODS = [
    "persistence", "same_time_last_week", "seasonal_average", "linear_regression",
    "spline_smoothing", "spline_hybrid", "decision_tree",
]


@dataclass
class FinalModels:
    """The fitted final models for one horizon.

    Attributes:
        horizon_minutes: Forecast horizon the models were built for.
        features: Feature column names, in model order.
        best_params: Chosen tree settings (``max_depth``, ``min_samples_leaf``).
        tree: Decision tree refitted on training plus validation rows.
        linear: Linear regression refitted on training plus validation rows.
        seasonal: Seasonal-average baseline refitted on training plus validation rows.
        profile: Smoothing-spline time-of-day profile.
        hybrid: Spline hybrid (profile plus persistence of today's deviation).
        tuning: The tuner's results table (from fits on the training subsample).
        train_rows: Number of training rows used for tuning.
        val_rows: Number of validation rows used for tuning.
        feature_ranges: ``{feature: (min, max)}`` over the rows the models were fitted on,
            used to flag forecasts that fall outside the training range.
    """

    horizon_minutes: int
    features: list
    best_params: dict
    tree: object
    linear: object
    seasonal: SeasonalAverage
    profile: SplineProfile
    hybrid: SplineAnomalyForecaster
    tuning: pd.DataFrame
    train_rows: int
    val_rows: int
    feature_ranges: dict = field(default_factory=dict)


def fit_final_models(
    train: pd.DataFrame,
    val: pd.DataFrame,
    horizon_minutes: int,
    stride: int = 4,
    tolerance: float = 0.005,
    spline_pad_hours: float = 3.0,
) -> FinalModels:
    """Tune on train/validation, then refit every learned method on both.

    Args:
        train: Training rows from ``chronological_split``.
        val: Validation rows (used to choose the tree's settings).
        horizon_minutes: Forecast horizon, recorded in the result.
        stride: Subsampling stride for tuning (see ``tune_decision_tree_fast``).
        tolerance: Simplest-tree tolerance for tuning.
        spline_pad_hours: Midnight wrap-around for the spline profile (3 h was measured to
            match the full 24 h wrap).

    Returns:
        A :class:`FinalModels`. The test set is not an input, so it cannot influence anything.

    Raises:
        ValueError: If ``train`` or ``val`` is empty.
    """
    if train.empty or val.empty:
        raise ValueError("train and val must be non-empty")
    features = feature_names()
    tuning, best = tune_decision_tree_fast(train, val, features, stride=stride, tolerance=tolerance)
    full = pd.concat([train, val], ignore_index=True)

    tree = fit_decision_tree(full, features, **best)
    linear = fit_linear(full, features)
    seasonal = SeasonalAverage().fit(full)
    profile = SplineProfile("smoothing", pad_hours=spline_pad_hours).fit(full)
    hybrid = SplineAnomalyForecaster(method="smoothing", pad_hours=spline_pad_hours).fit(full, profile=profile)
    ranges = {f: (float(full[f].min()), float(full[f].max())) for f in features}
    return FinalModels(
        horizon_minutes=horizon_minutes, features=features, best_params=best, tree=tree, linear=linear,
        seasonal=seasonal, profile=profile, hybrid=hybrid, tuning=tuning,
        train_rows=len(train), val_rows=len(val), feature_ranges=ranges,
    )


def predict_methods(models: FinalModels, df: pd.DataFrame) -> pd.DataFrame:
    """Predict with every final method (and baseline) for the given rows.

    Args:
        models: Result of :func:`fit_final_models`.
        df: Rows from ``build_supervised_dataset`` (features, ``target_timestamp``,
            ``baseline_last_week``).

    Returns:
        DataFrame, same row order as ``df``, with one column per entry of ``FINAL_METHODS``.
        ``same_time_last_week`` is NaN where last week's value is unavailable.
    """
    return pd.DataFrame({
        "persistence": predict_persistence(df),
        "same_time_last_week": predict_last_week(df),
        "seasonal_average": models.seasonal.predict(df),
        "linear_regression": models.linear.predict(df[models.features]),
        "spline_smoothing": models.profile.predict(df),
        "spline_hybrid": models.hybrid.predict(df),
        "decision_tree": models.tree.predict(df[models.features]),
    }, index=df.index)


def lookahead_flags(clean_df: pd.DataFrame, bin_minutes: int = 5, window_minutes: int = 60) -> pd.DataFrame:
    """Mark rows whose recent history contains values that used future information.

    A bin is "look-ahead" if it was interpolated (uses the next observed value) or capped as a
    spike (the spike rule uses a window that looks ahead). A forecast at time ``t`` is
    affected if any bin in ``[t - window, t]`` is look-ahead.

    Args:
        clean_df: Output of ``clean_occupancy`` (needs ``is_spike`` and ``fill_status``).
        bin_minutes: Grid spacing in minutes.
        window_minutes: History window ending at ``t`` (multiple of ``bin_minutes``); 60
            matches the longest lag and rolling window used as features.

    Returns:
        DataFrame with ``floorspaceid``, ``timestamp`` and ``uses_lookahead`` (bool).

    Raises:
        ValueError: If columns are missing or the minute arguments are invalid.
    """
    missing = {"floorspaceid", "timestamp", "is_spike", "fill_status"} - set(clean_df.columns)
    if missing:
        raise ValueError(f"clean_df is missing required columns: {sorted(missing)}")
    for name, v in (("bin_minutes", bin_minutes), ("window_minutes", window_minutes)):
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise ValueError(f"{name} must be a positive integer, got {v!r}")
    if window_minutes % bin_minutes:
        raise ValueError("window_minutes must be a multiple of bin_minutes")
    n = window_minutes // bin_minutes + 1  # bins t-window .. t inclusive

    df = clean_df.sort_values(["floorspaceid", "timestamp"]).reset_index(drop=True)
    flag = (df["is_spike"].astype(bool) | (df["fill_status"] == "interpolated")).astype(float)
    rolled = flag.groupby(df["floorspaceid"]).transform(lambda s: s.rolling(n, min_periods=1).max())
    return pd.DataFrame({"floorspaceid": df["floorspaceid"], "timestamp": df["timestamp"],
                         "uses_lookahead": rolled.to_numpy() > 0})


def block_bootstrap(
    y_true,
    predictions: dict,
    blocks,
    reference: str = "persistence",
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Bootstrap confidence intervals for RMSE and for improvement over a reference method.

    Whole blocks (e.g. room-days) are resampled with replacement, not individual rows, because
    errors at neighbouring times are strongly correlated; resampling rows would make the
    intervals look far narrower than they really are.

    Args:
        y_true: True values (people).
        predictions: Mapping of method name to predictions (finite, same length as ``y_true``).
        blocks: Block label per row (e.g. ``room + date``).
        reference: Method the others are compared with.
        n_boot: Number of bootstrap resamples.
        level: Confidence level, strictly between 0 and 1.
        seed: Random seed (results are reproducible).

    Returns:
        DataFrame indexed by method with ``rmse`` (point estimate) and its interval
        (``rmse_lo``, ``rmse_hi``); ``improvement_pct`` (``(1 - rmse / reference rmse) * 100``)
        with ``improvement_lo``, ``improvement_hi``; and ``prob_better`` (share of resamples in
        which the method beat the reference; 0 for the reference itself).

    Raises:
        ValueError: If inputs are inconsistent, contain NaN, or ``n_boot``/``level`` are invalid.
    """
    if isinstance(n_boot, bool) or not isinstance(n_boot, int) or n_boot < 1:
        raise ValueError(f"n_boot must be a positive integer, got {n_boot!r}")
    if not 0 < level < 1:
        raise ValueError(f"level must be strictly between 0 and 1, got {level!r}")
    if reference not in predictions:
        raise ValueError(f"reference {reference!r} is not among the predictions")
    y = np.asarray(y_true, dtype=float)
    labels = np.asarray(blocks)
    if len(labels) != len(y):
        raise ValueError("blocks must have the same length as y_true")
    preds = {k: np.asarray(v, dtype=float) for k, v in predictions.items()}
    for name, p in preds.items():
        if p.shape != y.shape:
            raise ValueError(f"predictions for {name!r} have the wrong length")
        if not np.isfinite(p).all():
            raise ValueError(f"predictions for {name!r} contain NaN or infinite values")
    if not np.isfinite(y).all():
        raise ValueError("y_true contains NaN or infinite values")

    codes, uniques = pd.factorize(labels)
    n_blocks = len(uniques)
    count = np.bincount(codes, minlength=n_blocks).astype(float)
    sse = {k: np.bincount(codes, weights=(p - y) ** 2, minlength=n_blocks) for k, p in preds.items()}

    draws = np.random.default_rng(seed).integers(0, n_blocks, size=(n_boot, n_blocks))
    denom = count[draws].sum(axis=1)
    rmse_boot = {k: np.sqrt(s[draws].sum(axis=1) / denom) for k, s in sse.items()}
    rmse_point = {k: float(np.sqrt(s.sum() / count.sum())) for k, s in sse.items()}

    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    rows = {}
    for k in preds:
        impr_boot = (1 - rmse_boot[k] / rmse_boot[reference]) * 100
        rows[k] = {
            "rmse": rmse_point[k],
            "rmse_lo": float(np.percentile(rmse_boot[k], lo)), "rmse_hi": float(np.percentile(rmse_boot[k], hi)),
            "improvement_pct": (1 - rmse_point[k] / rmse_point[reference]) * 100,
            "improvement_lo": float(np.percentile(impr_boot, lo)), "improvement_hi": float(np.percentile(impr_boot, hi)),
            "prob_better": float((impr_boot > 0).mean()),
        }
    return pd.DataFrame(rows).T
