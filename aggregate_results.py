"""Aggregate completed experiment directories into run and mean/std tables."""

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_GROUP_COLUMNS = [
    "experiment_name",
    "model",
    "variant",
    "latent_shape",
    "effective_latent_size",
    "loss_name",
    "split_seed",
]


def collect_results(output_dir):
    rows = []
    for result_path in sorted(Path(output_dir).glob("*/result.json")):
        with result_path.open("r", encoding="utf-8") as handle:
            row = json.load(handle)
        row["run_dir"] = str(result_path.parent)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No */result.json files found below {output_dir}")
    return pd.DataFrame(rows)


def aggregate_mean_std(runs, group_columns=None):
    group_columns = group_columns or DEFAULT_GROUP_COLUMNS
    group_columns = [column for column in group_columns if column in runs.columns]
    metric_columns = [
        column
        for column in runs.columns
        if column.startswith("best_val_") or column.startswith("final_val_")
    ]
    summary = runs.groupby(group_columns, dropna=False)[metric_columns].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--runs-name", default="all_runs.csv")
    parser.add_argument("--summary-name", default="mean_std.csv")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    runs = collect_results(output_dir)
    summary = aggregate_mean_std(runs)
    runs.to_csv(output_dir / args.runs_name, index=False)
    summary.to_csv(output_dir / args.summary_name, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
