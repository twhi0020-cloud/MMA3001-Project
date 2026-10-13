"""Baseline predictors and the decision tree regressor used for headcount forecasting.

Baselines (simplest first), all predicting the headcount ``h`` minutes ahead:

* **Persistence** - "nothing changes": predict the current headcount.
* **Same time last week** - the headcount at the target time one week earlier.
* **Seasonal average** - the mean headcount seen in the *training* data for the same room,
  day of week and time of day. It has no learning algorithm beyond averaging, so it shows
  how much of the pattern is simple repetition.
* **Linear regression** - a taught method on the same features as the tree.

The main model is a single :class:`sklearn.tree.DecisionTreeRegressor`, tuned over
``max_depth`` and ``min_samples_leaf`` by comparing training and validation RMSE (the
workflow from the unit's decision-tree notes). Variable importance uses permutation
importance on validation data.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def predict_persistence(df: pd.DataFrame) -> np.ndarray:
    """Predict that headcount stays at its current value (column ``lag_0m``).

    Args:
        df: Dataset from ``build_supervised_dataset`` (needs ``lag_0m``).

    Returns:
        Predicted headcount (people) for each row.

    Raises:
        ValueError: If ``lag_0m`` is missing.
    """
    if "lag_0m" not in df.columns:
        raise ValueError("df must contain 'lag_0m' (include lag 0 in lags_minutes)")
    return df["lag_0m"].to_numpy(dtype=float)


def predict_last_week(df: pd.DataFrame) -> np.ndarray:
    """Predict the headcount seen at the target time one week earlier.

    Args:
        df: Dataset from ``build_supervised_dataset`` (needs ``baseline_last_week``).

    Returns:
        Predicted headcount (people); NaN where last week's value is unavailable.

    Raises:
        ValueError: If ``baseline_last_week`` is missing.
    """
    if "baseline_last_week" not in df.columns:
        raise ValueError("df must contain 'baseline_last_week'")
    return df["baseline_last_week"].to_numpy(dtype=float)


class SeasonalAverage:
    """Mean training headcount for a room, day of week and time-of-day slot.

    Lookup falls back from (room, day, slot) to (room, slot) to the overall mean, so
    unseen combinations still get a prediction.

    Args:
        slot_minutes: Width of the time-of-day slots in minutes (divides 1440).

    Raises:
        ValueError: If ``slot_minutes`` is not a positive divisor of 1440.
    """

    def __init__(self, slot_minutes: int = 15):
        if isinstance(slot_minutes, bool) or not isinstance(slot_minutes, int) \
                or slot_minutes <= 0 or 1440 % slot_minutes:
            raise ValueError(f"slot_minutes must be a positive divisor of 1440, got {slot_minutes!r}")
        self.slot_minutes = slot_minutes
        self._full = None
        self._room_slot = None
        self._global_mean = None

    def _keys(self, df: pd.DataFrame) -> pd.DataFrame:
        tt = df["target_timestamp"]
        slot = (tt.dt.hour * 60 + tt.dt.minute) // self.slot_minutes
        return pd.DataFrame({
            "room": df["floorspaceid"].to_numpy(),
            "dow": tt.dt.dayofweek.to_numpy(),
            "slot": slot.to_numpy(),
        })

    def fit(self, train: pd.DataFrame) -> "SeasonalAverage":
        """Compute the seasonal means from training data only.

        Args:
            train: Training rows with ``floorspaceid``, ``target_timestamp`` and ``target``.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``train`` is empty.
        """
        if train.empty:
            raise ValueError("cannot fit on an empty training set")
        k = self._keys(train)
        k["y"] = train["target"].to_numpy(dtype=float)
        self._full = k.groupby(["room", "dow", "slot"])["y"].mean()
        self._room_slot = k.groupby(["room", "slot"])["y"].mean()
        self._global_mean = float(k["y"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict headcount (people) for each row from the fitted seasonal means.

        Args:
            df: Rows with ``floorspaceid`` and ``target_timestamp``.

        Returns:
            Array of predictions.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self._full is None:
            raise RuntimeError("SeasonalAverage must be fitted before predict")
        k = self._keys(df)
        pred = self._full.reindex(pd.MultiIndex.from_frame(k[["room", "dow", "slot"]])).to_numpy()
        fallback = self._room_slot.reindex(pd.MultiIndex.from_frame(k[["room", "slot"]])).to_numpy()
        pred = np.where(np.isnan(pred), fallback, pred)
        return np.where(np.isnan(pred), self._global_mean, pred)


def fit_linear(train: pd.DataFrame, features: list[str]) -> LinearRegression:
    """Fit ordinary least squares on the given features.

    Args:
        train: Training rows with the feature columns and ``target``.
        features: Feature column names.

    Returns:
        The fitted model.

    Raises:
        ValueError: If ``train`` is empty.
    """
    if train.empty:
        raise ValueError("cannot fit on an empty training set")
    return LinearRegression().fit(train[features], train["target"])


def fit_decision_tree(
    train: pd.DataFrame,
    features: list[str],
    max_depth: int | None,
    min_samples_leaf: int,
    random_state: int = 0,
) -> DecisionTreeRegressor:
    """Fit one decision tree regressor.

    Args:
        train: Training rows with the feature columns and ``target``.
        features: Feature column names.
        max_depth: Maximum tree depth (``None`` = grow until leaves are pure or too small).
        min_samples_leaf: Minimum training rows in each leaf.
        random_state: Seed, so ties between equally good splits resolve repeatably.

    Returns:
        The fitted tree.

    Raises:
        ValueError: If ``train`` is empty.
    """
    if train.empty:
        raise ValueError("cannot fit on an empty training set")
    model = DecisionTreeRegressor(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state
    )
    return model.fit(train[features], train["target"])


def _evaluate_settings(train, val, features, settings, random_state) -> pd.DataFrame:
    """Fit one tree per (max_depth, min_samples_leaf) pair and score it on validation."""
    x_tr, y_tr = train[features], train["target"].to_numpy(dtype=float)
    x_va, y_va = val[features], val["target"].to_numpy(dtype=float)
    rows = []
    for depth, leaf in settings:
        t0 = perf_counter()
        model = DecisionTreeRegressor(
            max_depth=depth, min_samples_leaf=leaf, random_state=random_state
        ).fit(x_tr, y_tr)
        fit_seconds = perf_counter() - t0
        err_va = model.predict(x_va) - y_va
        rows.append({
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "n_leaves": int(model.get_n_leaves()),
            "train_rmse": float(np.sqrt(np.mean((model.predict(x_tr) - y_tr) ** 2))),
            "val_rmse": float(np.sqrt(np.mean(err_va**2))),
            "val_mae": float(np.mean(np.abs(err_va))),
            "fit_seconds": fit_seconds,
        })
    return pd.DataFrame(rows)


def _select_best(results: pd.DataFrame, tolerance: float) -> tuple[pd.DataFrame, dict]:
    """Mark the simplest tree within ``tolerance`` of the best validation RMSE."""
    results = results.reset_index(drop=True)
    eligible = results[results["val_rmse"] <= results["val_rmse"].min() * (1 + tolerance)]
    best_idx = eligible.sort_values(["n_leaves", "val_rmse"]).index[0]
    results["selected"] = results.index == best_idx
    best_row = results.loc[best_idx]
    depth = best_row["max_depth"]
    best = {
        "max_depth": None if pd.isna(depth) else int(depth),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
    }
    return results, best


def tune_decision_tree(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    max_depths: list,
    min_samples_leaf_values: list,
    random_state: int = 0,
    tolerance: float = 0.005,
) -> tuple[pd.DataFrame, dict]:
    """Grid-search tree complexity, comparing training and validation error.

    Overfitting shows up as training RMSE falling while validation RMSE stops improving
    or rises. Among settings whose validation RMSE is within ``tolerance`` (a fraction) of
    the best, the one with the fewest leaves is chosen: differences smaller than the
    validation noise should not decide between a small and a large tree.

    Args:
        train: Training rows.
        val: Validation rows (never used to fit).
        features: Feature column names.
        max_depths: ``max_depth`` values to try (``None`` allowed).
        min_samples_leaf_values: ``min_samples_leaf`` values to try.
        random_state: Seed passed to each tree.
        tolerance: Allowed relative increase in validation RMSE over the best setting
            (0.005 = 0.5 %). Use 0 to always pick the lowest validation RMSE.

    Returns:
        ``(results, best)``: ``results`` has one row per setting with ``max_depth``,
        ``min_samples_leaf``, ``n_leaves``, ``train_rmse``, ``val_rmse``, ``val_mae``,
        ``fit_seconds`` and ``selected`` (True for the chosen row); ``best`` is
        ``{"max_depth": ..., "min_samples_leaf": ...}``.

    Raises:
        ValueError: If a grid is empty, either dataset is empty, or ``tolerance`` is
            negative.
    """
    if not max_depths or not min_samples_leaf_values:
        raise ValueError("max_depths and min_samples_leaf_values must be non-empty")
    if train.empty or val.empty:
        raise ValueError("train and val must be non-empty")
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance!r}")

    settings = [(d, leaf) for d in max_depths for leaf in min_samples_leaf_values]
    return _select_best(_evaluate_settings(train, val, features, settings, random_state), tolerance)


def tune_decision_tree_fast(
    train: pd.DataFrame,
    val: pd.DataFrame,
    features: list[str],
    stride: int = 4,
    coarse_depths: tuple = (4, 8, 12),
    coarse_leaves: tuple = (10, 200),
    random_state: int = 0,
    tolerance: float = 0.005,
) -> tuple[pd.DataFrame, dict]:
    """Find a good tree with far fewer, cheaper fits than an exhaustive grid.

    Two ideas, each measured against the exhaustive search (see the profiling results):

    1. **Subsample:** tune on every ``stride``-th training row. Tree fit time grows roughly
       linearly with rows, while validation error flattens out well before all the data is
       used, so a quarter of the rows usually selects the same tree.
    2. **Coarse to fine:** try a small coarse grid, then a 3 x 3 neighbourhood around the
       winner (depth +/- 2, leaf size / 4, x 1, x 4). If the winner sits on the edge of the
       coarse grid the neighbourhood extends beyond it.

    The returned settings come from fits on the *subsample*; refit the chosen tree on the
    full ``train`` with :func:`fit_decision_tree`. ``val`` is never subsampled.

    Args:
        train: Training rows.
        val: Validation rows (never used to fit).
        features: Feature column names.
        stride: Use every ``stride``-th training row for tuning (1 = all rows).
        coarse_depths: Stage-1 ``max_depth`` values (integers).
        coarse_leaves: Stage-1 ``min_samples_leaf`` values (integers).
        random_state: Seed passed to each tree.
        tolerance: Simplest-tree tolerance, as in :func:`tune_decision_tree`.

    Returns:
        ``(results, best)`` as for :func:`tune_decision_tree`, with an extra ``stage`` column
        (1 = coarse, 2 = refinement). No setting is fitted twice.

    Raises:
        ValueError: If ``stride`` is not a positive integer, a coarse grid is empty or contains
            non-integers, either dataset is empty, or ``tolerance`` is negative.
    """
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError(f"stride must be a positive integer, got {stride!r}")
    if not coarse_depths or not coarse_leaves:
        raise ValueError("coarse_depths and coarse_leaves must be non-empty")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (*coarse_depths, *coarse_leaves)):
        raise ValueError("coarse grid values must be positive integers")
    if train.empty or val.empty:
        raise ValueError("train and val must be non-empty")
    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance!r}")

    sub = train.iloc[::stride]
    stage1 = [(d, leaf) for d in coarse_depths for leaf in coarse_leaves]
    r1 = _evaluate_settings(sub, val, features, stage1, random_state)
    _, best1 = _select_best(r1, tolerance)

    d0, l0 = best1["max_depth"], best1["min_samples_leaf"]
    depths = sorted({max(2, d0 - 2), d0, d0 + 2})
    leaves = sorted({max(1, l0 // 4), l0, l0 * 4})
    stage2 = [(d, leaf) for d in depths for leaf in leaves if (d, leaf) not in set(stage1)]
    r2 = _evaluate_settings(sub, val, features, stage2, random_state)

    results = pd.concat([r1.assign(stage=1), r2.assign(stage=2)], ignore_index=True)
    return _select_best(results, tolerance)


def permutation_importance_table(
    model,
    df: pd.DataFrame,
    features: list[str],
    n_repeats: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Rank features by how much shuffling each one increases RMSE.

    Computed on data the model was not trained on, so it reflects what the model relies
    on for forecasting rather than what it memorised.

    Args:
        model: A fitted regressor.
        df: Rows (normally the validation set) with the feature columns and ``target``.
        features: Feature column names.
        n_repeats: Number of shuffles per feature.
        random_state: Seed for the shuffles.

    Returns:
        DataFrame sorted by ``rmse_increase`` (people) with columns ``feature``,
        ``rmse_increase`` and ``rmse_increase_std``.

    Raises:
        ValueError: If ``df`` is empty.
    """
    if df.empty:
        raise ValueError("df must be non-empty")
    result = permutation_importance(
        model, df[features], df["target"], n_repeats=n_repeats, random_state=random_state,
        scoring="neg_root_mean_squared_error",
    )
    table = pd.DataFrame({
        "feature": features,
        "rmse_increase": result.importances_mean,
        "rmse_increase_std": result.importances_std,
    })
    return table.sort_values("rmse_increase", ascending=False).reset_index(drop=True)
