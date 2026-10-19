"""Check that the live ``forecast_headcount`` path reproduces the batch (training) path on real data.

The batch path cleans the whole history at once; the live path sees only recent raw readings. They
can differ because gap interpolation and spike capping in the batch path use look-ahead. This script
samples validation rows (never test rows), rebuilds each forecast from the three hours of raw readings
before it, and compares the result with the batch prediction. It uses no test data and fits nothing.

Usage (from the repository root, after ``scripts/train_final.py``)::

    python scripts/check_live_vs_batch.py --n 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from occupancy_forecast.cleaning import clean_occupancy  # noqa: E402
from occupancy_forecast.features import build_supervised_dataset  # noqa: E402
from occupancy_forecast.forecast import forecast_headcount, load_bundle  # noqa: E402
from occupancy_forecast.validation import chronological_split  # noqa: E402

TZ = "Australia/Melbourne"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=300, help="validation rows to sample")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lookback-hours", type=int, nargs="+", default=[3, 24],
                   help="how much raw history is passed to the forecast (each is reported)")
    p.add_argument("--occupancy-csv", type=Path, default=ROOT / "data/raw/5occupancySensor_MayToDec2024_9MRows.csv")
    p.add_argument("--bundle", type=Path, default=ROOT / "models/final_h30.joblib")
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = p.parse_args(argv)

    bundle = load_bundle(args.bundle)
    h = bundle["horizon_minutes"]
    clean = clean_occupancy(pd.read_parquet(args.cache_dir / "binned_5min.parquet"))
    _, val, _ = chronological_split(build_supervised_dataset(clean, h))
    sample = val.sample(args.n, random_state=args.seed).reset_index(drop=True)
    sample["batch_pred"] = np.clip(bundle["model"].predict(sample[bundle["features"]]), 0, None)

    # one scan of the raw file for every (room, date) the samples need (and the day before, for midnight)
    need = pd.concat([sample[["floorspaceid", "timestamp"]],
                      sample[["floorspaceid", "timestamp"]].assign(timestamp=sample["timestamp"] - pd.Timedelta(days=1))])
    pairs = pd.DataFrame({"room": need["floorspaceid"], "d": need["timestamp"].dt.date}).drop_duplicates()
    con = duckdb.connect()
    con.register("pairs", pairs)
    raw = con.execute(
        f"""SELECT r.floorspaceid, r.headcount, r.collecteddate
            FROM read_csv_auto('{args.occupancy_csv}') r
            JOIN pairs p ON r.floorspaceid = p.room
             AND CAST(r.collecteddate AT TIME ZONE '{TZ}' AS DATE) = p.d"""
    ).fetchdf()
    raw["local"] = raw["collecteddate"].dt.tz_convert(TZ).dt.tz_localize(None)
    print(f"Loaded {len(raw):,} raw readings for {len(pairs)} room-days", flush=True)

    by_room = {room: g for room, g in raw.groupby("floorspaceid")}
    rmse = lambda pred, m, frame: float(np.sqrt(np.mean((pred[m] - frame.loc[m, "target"]) ** 2)))
    columns, kept = {}, {}
    for hours in args.lookback_hours:
        live, status = [], []
        for _, row in sample.iterrows():
            t = row["timestamp"]
            g = by_room[row["floorspaceid"]]
            window = g[(g["local"] >= t - pd.Timedelta(hours=hours)) & (g["local"] < t + pd.Timedelta(minutes=5))]
            if window.empty:
                live.append(np.nan)
                status.append("no_readings")
                continue
            out = forecast_headcount(window[["floorspaceid", "headcount", "collecteddate"]], bundle, as_of=t)
            live.append(out.loc[0, "predicted_headcount"])
            status.append(out.loc[0, "status"])
        frame = sample.assign(live_pred=live, status=status)
        ok = frame["status"] == "ok"
        diff = (frame["live_pred"] - frame["batch_pred"])[ok]
        columns[f"lookback_{hours}h"] = {
            "rows_sampled": len(frame),
            "live_forecast_available_%": ok.mean() * 100,
            "unavailable_insufficient_history_%": (frame["status"] == "insufficient_history").mean() * 100,
            "unavailable_no_readings_in_window_%": (frame["status"] == "no_readings").mean() * 100,
            "identical_to_batch_%_of_available": float((diff.abs() < 1e-9).mean() * 100),
            "within_0.1_people_%_of_available": float((diff.abs() < 0.1).mean() * 100),
            "max_abs_difference_people": float(diff.abs().max()),
            "rmse_of_difference_people": float(np.sqrt((diff**2).mean())),
            "rmse_vs_truth_batch_same_rows": rmse(frame["batch_pred"], ok, frame),
            "rmse_vs_truth_live_same_rows": rmse(frame["live_pred"], ok, frame),
        }
        kept[hours] = (frame, ok, diff)
    table = pd.DataFrame(columns).rename_axis("metric")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.results_dir / "live_vs_batch.csv")
    print("\nLive forecast_headcount() vs batch pipeline, validation rows only:")
    print(table.round(3).to_string())
    sample, ok, diff = kept[args.lookback_hours[-1]][0], kept[args.lookback_hours[-1]][1], kept[args.lookback_hours[-1]][2]
    bad = sample[ok & ((sample["live_pred"] - sample["batch_pred"]).abs() >= 0.1)]
    if len(bad):
        print(f"\n{len(bad)} rows differ by 0.1 or more; their fill status at the target and first few examples:")
        print(bad[["floorspaceid", "timestamp", "batch_pred", "live_pred"]].head(5).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
