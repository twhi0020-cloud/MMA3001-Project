"""Train the final model for one horizon and save it as a bundle for ``forecast_headcount``.

The tree's settings are chosen with the fast tuner (training rows, scored on validation rows);
every learned method is then refitted on training plus validation rows. The test rows are not
touched. Usage (from the repository root, with the cache from ``scripts/train.py`` present)::

    python scripts/train_final.py --horizon 30
    python scripts/train_final.py --horizon 60 --out models/final_h60.joblib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occupancy_forecast.cleaning import clean_occupancy  # noqa: E402
from occupancy_forecast.features import build_supervised_dataset  # noqa: E402
from occupancy_forecast.final import fit_final_models  # noqa: E402
from occupancy_forecast.forecast import make_bundle, save_bundle  # noqa: E402
from occupancy_forecast.validation import chronological_split  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--horizon", type=int, default=30, help="forecast horizon in minutes")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed")
    p.add_argument("--out", type=Path, default=None, help="where to save the bundle")
    args = p.parse_args(argv)
    out = args.out or ROOT / "models" / f"final_h{args.horizon}.joblib"

    cache = args.cache_dir / "binned_5min.parquet"
    if not cache.is_file():
        sys.exit(f"Missing {cache}; run scripts/train.py first to create it.")
    t0 = perf_counter()
    clean = clean_occupancy(pd.read_parquet(cache))
    train, val, _ = chronological_split(build_supervised_dataset(clean, args.horizon))
    models = fit_final_models(train, val, args.horizon)
    path = save_bundle(make_bundle(models), out)
    print(f"Horizon {args.horizon} min: tuned on {models.train_rows:,} train / {models.val_rows:,} validation rows")
    print(f"Chosen tree: {models.best_params}, {models.tree.get_n_leaves()} leaves, "
          f"refitted on {models.tree.tree_.n_node_samples[0]:,} rows")
    print(f"Saved {path} in {perf_counter() - t0:.0f} s. The test set was not used.")


if __name__ == "__main__":
    main()
