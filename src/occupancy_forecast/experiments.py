"""Run the forecasting pipeline under different settings and compare the results fairly.

The sensitivity analysis asks how much the conclusions depend on choices that cannot be
verified against ground truth (the ``assumed_empty`` gap rule, the bin width, the horizon).
Each choice is one *condition*: clean the data, build the supervised table, split, fit the
baselines and a tuned decision tree, and keep the **validation** predictions. The test set
is never touched here.

Fair comparison needs two things, both provided here:

* the validation window is fixed once (:func:`reference_boundaries`) and reused, so
  conditions that keep different rows still evaluate over the same calendar period;
* :func:`compare_conditions` scores every condition on the *identical* rows (those valid in
  all conditions, optionally only where the target was actually observed). On those rows the
  features and targets are identical across conditions, so a difference in score comes only
  from how the training data was prepared.

Absolute errors are **not** comparable between different horizons or bin widths, because the
quantity being predicted changes. For those, compare each method's improvement over
persistence instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import pandas as pd

from occupancy_forecast.cleaning import clean_occupancy
from occupancy_forecast.features import build_supervised_dataset, feature_names
from occupancy_forecast.models import (
    SeasonalAverage,
    fit_decision_tree,
    fit_linear,
    predict_last_week,
    predict_persistence,
    tune_decision_tree,
)
from occupancy_forecast.splines import SPLINE_METHODS, SplineAnomalyForecaster, SplineProfile
from occupancy_forecast.validation import (
    chronological_split,
    compare_predictions,
    split_boundaries,
)

METHODS = ["persistence", "same_time_last_week", "seasonal_average", "linear_regression", "decision_tree"]
SPLINE_COLUMNS = [f"spline_{m}" for m in SPLINE_METHODS] + ["spline_hybrid"]
KEY = ["floorspaceid", "timestamp"]


@dataclass
class ConditionResult:
    """Outcome of one experimental condition (validation data only).

    Attributes:
        name: Label of the condition.
        bin_minutes: Bin width used.
        horizon_minutes: Forecast horizon used.
        n_train: Number of training rows.
        n_val: Number of validation rows.
        best_params: Chosen tree settings.
        tuning: Tuning table from :func:`occupancy_forecast.models.tune_decision_tree`.
        val: One row per validation example with ``floorspaceid``, ``timestamp``,
            ``target``, ``target_fill_status`` and a ``pred_<method>`` column per method.
        seconds: Wall-clock seconds per stage (``clean``, ``features``, ``tuning`` and, if
            splines were fitted, ``splines``).
        spline_diagnostics: Per-curve summaries (including each curve's minimum) for every
            spline method, or ``None`` if splines were not fitted.
        spline_phi: Fitted persistence fraction ``phi`` of the spline hybrid (how much of the
            current deviation from the normal profile is expected to remain at the target
            time), or ``None`` if splines were not fitted.
    """

    name: str
    bin_minutes: int
    horizon_minutes: int
    n_train: int
    n_val: int
    best_params: dict
    tuning: pd.DataFrame
    val: pd.DataFrame
    seconds: dict = field(default_factory=dict)
    spline_diagnostics: pd.DataFrame | None = None
    spline_phi: float | None = None


def scaled_clean_kwargs(bin_minutes: int, **overrides) -> dict:
    """Cleaning settings that cover the same *durations* at any bin width.

    The defaults of ``clean_occupancy`` are counted in 5-minute bins. Changing the bin
    width without rescaling would silently change what they mean (e.g. 3 bins = 15 min at
    5 min but 45 min at 15 min). This keeps them time-equivalent: a 245-minute spike window,
    40 minutes of required history, 15-minute interpolation limit and 1-day zero-fill limit.

    Args:
        bin_minutes: Bin width in minutes (positive integer, at most 15).
        **overrides: Any ``clean_occupancy`` keyword to override after scaling
            (e.g. ``quiet_threshold=1.0`` or ``max_quiet_bins=0``).

    Returns:
        Keyword arguments for :func:`occupancy_forecast.cleaning.clean_occupancy`.

    Raises:
        ValueError: If ``bin_minutes`` is not an integer between 1 and 15.
    """
    if isinstance(bin_minutes, bool) or not isinstance(bin_minutes, int) or not 1 <= bin_minutes <= 15:
        raise ValueError(f"bin_minutes must be an integer between 1 and 15, got {bin_minutes!r}")
    window = max(3, round(245 / bin_minutes))
    if window % 2 == 0:
        window += 1
    kwargs = {
        "bin_minutes": bin_minutes,
        "window_bins": window,
        "min_periods": max(2, round(40 / bin_minutes)),
        "max_interp_bins": max(1, 15 // bin_minutes),
        "max_quiet_bins": 1440 // bin_minutes,
    }
    kwargs.update(overrides)
    return kwargs


def reference_boundaries(
    binned: pd.DataFrame,
    horizon_minutes: int = 30,
    bin_minutes: int = 5,
    clean_overrides: dict | None = None,
) -> dict:
    """Compute the per-room validation window of the reference condition.

    Args:
        binned: Output of ``resample_occupancy``.
        horizon_minutes: Horizon of the reference condition.
        bin_minutes: Bin width of the reference condition.
        clean_overrides: Optional cleaning overrides for the reference condition.

    Returns:
        Per-room ``(b1, b2)`` cut times, as from
        :func:`occupancy_forecast.validation.split_boundaries`.
    """
    clean = clean_occupancy(binned, **scaled_clean_kwargs(bin_minutes, **(clean_overrides or {})))
    data = build_supervised_dataset(clean, horizon_minutes=horizon_minutes, bin_minutes=bin_minutes)
    return split_boundaries(data)


def run_condition(
    binned: pd.DataFrame,
    name: str,
    horizon_minutes: int = 30,
    bin_minutes: int = 5,
    clean_overrides: dict | None = None,
    boundaries: dict | None = None,
    depths: tuple = (6, 8, 10, 12),
    leaves: tuple = (50, 200),
    tolerance: float = 0.005,
    splines: bool = False,
    spline_config: dict | None = None,
    fit_tree: bool = True,
) -> ConditionResult:
    """Clean, build features, split, fit all methods and return validation predictions.

    Args:
        binned: Output of ``resample_occupancy`` at ``bin_minutes``.
        name: Label for the condition.
        horizon_minutes: Forecast horizon in minutes.
        bin_minutes: Bin width the data was resampled at.
        clean_overrides: Cleaning overrides on top of :func:`scaled_clean_kwargs`.
        boundaries: Fixed validation window from :func:`reference_boundaries`. If ``None``,
            the condition's own 60/20/20 split is used.
        depths: ``max_depth`` values for tree tuning.
        leaves: ``min_samples_leaf`` values for tree tuning.
        tolerance: Simplest-tree tolerance (see ``tune_decision_tree``).
        splines: Also fit the time-of-day spline profiles and the spline hybrid, adding
            ``pred_spline_<method>`` and ``pred_spline_hybrid`` columns.
        spline_config: Options for :class:`~occupancy_forecast.splines.SplineProfile`
            (``slot_minutes``, ``day_grouping``); the key ``hybrid_method`` (default
            ``"smoothing"``) picks the spline used inside the hybrid.
        fit_tree: If ``False``, skip tree tuning (faster; no ``pred_decision_tree`` column).

    Returns:
        A :class:`ConditionResult`. The test set is not evaluated.

    Raises:
        ValueError: If the resulting training or validation set is empty.
    """
    seconds = {}
    t0 = perf_counter()
    clean = clean_occupancy(binned, **scaled_clean_kwargs(bin_minutes, **(clean_overrides or {})))
    seconds["clean"] = perf_counter() - t0

    t0 = perf_counter()
    data = build_supervised_dataset(clean, horizon_minutes=horizon_minutes, bin_minutes=bin_minutes)
    train, val, _ = chronological_split(data, boundaries=boundaries)
    seconds["features"] = perf_counter() - t0
    if train.empty or val.empty:
        raise ValueError(f"condition {name!r} produced an empty train or validation set")

    features = feature_names()
    t0 = perf_counter()
    seasonal = SeasonalAverage().fit(train)
    linear = fit_linear(train, features)
    if fit_tree:
        tuning, best = tune_decision_tree(train, val, features, list(depths), list(leaves), tolerance=tolerance)
        tree = fit_decision_tree(train, features, **best)
    else:
        tuning, best, tree = pd.DataFrame(), {}, None
    seconds["tuning"] = perf_counter() - t0

    out = val[KEY + ["target", "target_fill_status"]].reset_index(drop=True)
    out["pred_persistence"] = predict_persistence(val)
    out["pred_same_time_last_week"] = predict_last_week(val)
    out["pred_seasonal_average"] = seasonal.predict(val)
    out["pred_linear_regression"] = linear.predict(val[features])
    if tree is not None:
        out["pred_decision_tree"] = tree.predict(val[features])

    diagnostics, phi = None, None
    if splines:
        t0 = perf_counter()
        cfg = dict(spline_config or {})
        hybrid_method = cfg.pop("hybrid_method", "smoothing")
        diag_parts, profiles = [], {}
        for m in SPLINE_METHODS:
            profiles[m] = SplineProfile(m, **cfg).fit(train)
            out[f"pred_spline_{m}"] = profiles[m].predict(val)
            diag_parts.append(profiles[m].diagnostics().assign(method=m))
        # the hybrid reuses the profile already fitted for its method (identical, and costly)
        hybrid = SplineAnomalyForecaster(method=hybrid_method, **cfg).fit(train, profile=profiles.get(hybrid_method))
        out["pred_spline_hybrid"] = hybrid.predict(val)
        phi = hybrid.phi_
        diagnostics = pd.concat(diag_parts, ignore_index=True)
        seconds["splines"] = perf_counter() - t0

    return ConditionResult(
        name=name, bin_minutes=bin_minutes, horizon_minutes=horizon_minutes,
        n_train=len(train), n_val=len(val), best_params=best, tuning=tuning, val=out, seconds=seconds,
        spline_diagnostics=diagnostics, spline_phi=phi,
    )


def compare_conditions(results: list, observed_only: bool = True, methods: list | None = None) -> pd.DataFrame:
    """Score every method in every condition on the identical validation rows.

    Only rows present in *all* conditions (and with a prediction from every method) are
    used. With ``observed_only`` the rows are further restricted to targets that were
    really measured, so that conditions which fill more gaps are not rewarded for easy
    filled-in zeros.

    Args:
        results: Conditions from :func:`run_condition` (all with the same horizon and bin
            width, otherwise the targets differ and the comparison is meaningless).
        observed_only: Keep only rows whose ``target_fill_status`` is ``observed``.
        methods: Methods to score (names without the ``pred_`` prefix). Defaults to
            :data:`METHODS`; pass e.g. ``METHODS + SPLINE_COLUMNS`` for spline conditions.

    Returns:
        DataFrame with a ``(condition, method)`` index and columns ``mae``, ``rmse``, ``n``.

    Raises:
        ValueError: If ``results`` is empty, the conditions differ in horizon or bin width,
            a requested method is missing from a condition, or no rows are common to all
            conditions.
    """
    if not results:
        raise ValueError("results must contain at least one condition")
    if len({(r.horizon_minutes, r.bin_minutes) for r in results}) > 1:
        raise ValueError("conditions must share the same horizon and bin width to be compared on identical rows")
    methods = list(methods) if methods else list(METHODS)
    for r in results:
        absent = [m for m in methods if f"pred_{m}" not in r.val.columns]
        if absent:
            raise ValueError(f"condition {r.name!r} has no predictions for: {absent}")

    frames = []
    for r in results:
        v = r.val.copy()
        if observed_only:
            v = v[v["target_fill_status"] == "observed"]
        frames.append(v.set_index(KEY))
    common = frames[0].index
    for f in frames[1:]:
        common = common.intersection(f.index)
    if len(common) == 0:
        raise ValueError("no validation rows are common to all conditions")

    pred_cols = [f"pred_{m}" for m in methods]
    aligned = [f.loc[common] for f in frames]
    # A row is used only if every method has a prediction in *every* condition; otherwise
    # conditions would silently be scored on different rows.
    valid = pd.Series(True, index=common)
    for f in aligned:
        valid &= f[pred_cols].notna().all(axis=1)
    if not valid.any():
        raise ValueError("no validation rows are common to all conditions")

    rows = {}
    for r, f in zip(results, aligned):
        f = f[valid]
        table = compare_predictions(f["target"].to_numpy(), {m: f[f"pred_{m}"].to_numpy() for m in methods})
        for method in methods:
            rows[(r.name, method)] = table.loc[method]
    out = pd.DataFrame(rows).T
    out.index.names = ["condition", "method"]
    return out.astype({"n": int})
