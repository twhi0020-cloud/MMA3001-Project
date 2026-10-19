"""The sealed test-set evaluation. Run it ONCE, after every modelling decision is final.

PROTOCOL (fixed before the test rows were first scored):

* Horizons: 15, 30 and 60 minutes. Data: the cleaned 5-minute grid with default settings.
* Split: each room's rows 60 % train / 20 % validation / 20 % test, by time, with an embargo.
* Models: persistence, same time last week, seasonal average, linear regression, smoothing-spline
  profile, spline hybrid and the decision tree. Tree settings come from the fast tuner (train rows,
  scored on validation rows); every learned method is then refitted on train + validation.
* Scoring: all methods on identical test rows. Three row sets are reported:
  ``observed`` (targets that were actually measured: the headline figure), ``all`` (also targets
  that were interpolated or assumed empty) and ``causal`` (observed targets whose previous hour
  contains no interpolated or capped bins, i.e. no look-ahead in the features).
* Uncertainty: 95 % intervals from a block bootstrap over whole room-days (2000 resamples).
* No model, setting or metric is changed after seeing these results. If a defect is found in
  this script, it must be reported together with the corrected numbers.

Usage (from the repository root)::

    python scripts/final_evaluation.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occupancy_forecast.cleaning import clean_occupancy  # noqa: E402
from occupancy_forecast.features import build_supervised_dataset  # noqa: E402
from occupancy_forecast.final import (  # noqa: E402
    FINAL_METHODS,
    block_bootstrap,
    fit_final_models,
    lookahead_flags,
    predict_methods,
)
from occupancy_forecast.validation import chronological_split, regression_metrics  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizons", type=int, nargs="+", default=[15, 30, 60])
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    cache = args.cache_dir / "binned_5min.parquet"
    if not cache.is_file():
        sys.exit(f"Missing {cache}; run scripts/train.py first to create it.")
    clean = clean_occupancy(pd.read_parquet(cache))
    flags = lookahead_flags(clean)

    main_rows, linear_rows, room_rows, size_rows = [], [], [], []
    for h in args.horizons:
        t0 = perf_counter()
        train, val, test = chronological_split(build_supervised_dataset(clean, h))
        models = fit_final_models(train, val, h)
        preds = predict_methods(models, test)

        test = test.merge(flags, on=["floorspaceid", "timestamp"], how="left")
        test["uses_lookahead"] = test["uses_lookahead"].fillna(False).astype(bool)
        observed = (test["target_fill_status"] == "observed").to_numpy()
        subsets = {
            "observed": observed,
            "all": np.ones(len(test), dtype=bool),
            "causal": observed & ~test["uses_lookahead"].to_numpy(),
        }
        size_rows.append({
            "horizon": h, "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
            "test_observed_targets": int(observed.sum()), "test_causal_rows": int(subsets["causal"].sum()),
            "share_rows_using_lookahead": float(test["uses_lookahead"].mean()),
            "max_depth": models.best_params["max_depth"], "min_samples_leaf": models.best_params["min_samples_leaf"],
            "n_leaves": int(models.tree.get_n_leaves()),
        })

        y_all = test["target"].to_numpy()
        blocks_all = (test["floorspaceid"] + "|" + test["timestamp"].dt.strftime("%Y-%m-%d")).to_numpy()
        for name, mask in subsets.items():
            common = mask & preds.notna().all(axis=1).to_numpy()
            y, b = y_all[common], blocks_all[common]
            p_sub = {m: preds.loc[common, m].to_numpy() for m in FINAL_METHODS}
            vs_persistence = block_bootstrap(y, p_sub, b, reference="persistence", n_boot=args.n_boot)
            vs_linear = block_bootstrap(y, p_sub, b, reference="linear_regression", n_boot=args.n_boot)
            for m in FINAL_METHODS:
                row = vs_persistence.loc[m].to_dict()
                row.update({"horizon": h, "rows": int(common.sum()), "subset": name, "method": m,
                            "mae": regression_metrics(y, p_sub[m])["mae"]})
                main_rows.append(row)
            lin = vs_linear.loc[["decision_tree", "spline_hybrid"]].reset_index(names="method")
            lin.insert(0, "subset", name)
            lin.insert(0, "horizon", h)
            linear_rows.append(lin)
            if name == "observed":
                rooms = test.loc[common, "floorspaceid"].str[:8].to_numpy()
                for room in sorted(set(rooms)):
                    r = rooms == room
                    row = {"horizon": h, "room": room, "rows": int(r.sum())}
                    for m in ("persistence", "linear_regression", "spline_hybrid", "decision_tree"):
                        row[f"rmse_{m}"] = regression_metrics(y[r], p_sub[m][r])["rmse"]
                    row["tree_vs_persistence_%"] = (1 - row["rmse_decision_tree"] / row["rmse_persistence"]) * 100
                    room_rows.append(row)
        print(f"  horizon {h} min scored in {perf_counter() - t0:.0f} s", flush=True)

    main_table = pd.DataFrame(main_rows)[["horizon", "subset", "method", "rows", "rmse", "rmse_lo", "rmse_hi", "mae",
                                          "improvement_pct", "improvement_lo", "improvement_hi", "prob_better"]]
    main_table.to_csv(args.results_dir / "final_test_main.csv", index=False)
    pd.concat(linear_rows).rename(columns={"improvement_pct": "improvement_vs_linear_pct"}).to_csv(
        args.results_dir / "final_test_vs_linear.csv", index=False)
    pd.DataFrame(room_rows).to_csv(args.results_dir / "final_test_by_room.csv", index=False)
    pd.DataFrame(size_rows).to_csv(args.results_dir / "final_test_sizes.csv", index=False)

    pd.set_option("display.width", 200)
    for h in args.horizons:
        t = main_table[(main_table["horizon"] == h) & (main_table["subset"] == "observed")]
        print(f"\nTEST SET, {h} min ahead, measured targets (RMSE in people; improvement vs persistence, 95% CI)")
        print(t.drop(columns=["horizon", "subset"]).round(3).to_string(index=False))
    print("\nTree vs linear regression (improvement %, 95% CI), observed targets")
    vs = pd.concat(linear_rows)
    print(vs[(vs["subset"] == "observed") & (vs["method"] == "decision_tree")].round(2).to_string(index=False))
    print("\nBy room (observed targets):")
    print(pd.DataFrame(room_rows).round(3).to_string(index=False))
    print("\nSizes and chosen trees:")
    print(pd.DataFrame(size_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
