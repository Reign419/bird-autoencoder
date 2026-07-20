"""Aggregate reconstruction, concept, intervention, and leakage results."""

import argparse
import json
from pathlib import Path

import pandas as pd


def summarize(table, groups, metrics):
    metrics = [column for column in metrics if column in table.columns]
    if not metrics:
        return pd.DataFrame()
    result = table.groupby(groups, dropna=False)[metrics].agg(["mean", "std", "count"])
    result.columns = [f"{metric}_{stat}" for metric, stat in result.columns]
    return result.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    args = parser.parse_args()
    root = Path(args.output_dir)
    run_rows, intervention_rows, concept_rows, semantic_rows, probe_rows = [], [], [], [], []

    for result_path in sorted(root.glob("*/result.json")):
        run_dir = result_path.parent
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["run_dir"] = str(run_dir)
        run_rows.append(result)
        identity = {
            "experiment_name": result.get("experiment_name"),
            "training_seed": result.get("training_seed"),
            "run_dir": str(run_dir),
        }
        for filename, destination in [
            ("group_interventions.csv", intervention_rows),
            ("concept_group_metrics.csv", concept_rows),
            ("semantic_bottleneck_analysis.csv", semantic_rows),
            ("concept_probe_groups.csv", probe_rows),
        ]:
            path = run_dir / filename
            if path.exists():
                table = pd.read_csv(path)
                for key, value in identity.items():
                    table[key] = value
                destination.extend(table.to_dict("records"))

    if not run_rows:
        raise FileNotFoundError(f"No */result.json files found under {root}")
    runs = pd.DataFrame(run_rows)
    runs.to_csv(root / "factorized_all_runs.csv", index=False)
    summarize(
        runs,
        ["experiment_name", "mode", "concept_dim"],
        [column for column in runs.columns if column.startswith("mean_")]
        + ["validation_macro_ap", "validation_macro_f1", "mean_group_u_global_ssim"],
    ).to_csv(root / "factorized_reconstruction_summary.csv", index=False)

    specifications = [
        (
            intervention_rows,
            "factorized_intervention_summary.csv",
            ["experiment_name", "group"],
            [
                "u_global_ssim_all",
                "u_global_ssim_effective",
                "mean_pixel_change_all",
                "mean_pixel_change_effective",
                "u_local_pixel_effective",
                "localization_ratio_effective",
                "pixel_change_p95_effective",
                "pixel_change_p99_effective",
                "bird_bbox_pixel_change_effective",
                "bird_outside_bbox_pixel_change_effective",
                "bird_bbox_enrichment_effective",
                "top1pct_in_bird_bbox_effective",
                "top1pct_bird_bbox_enrichment_effective",
                "local_non_target_pixel_effective",
                "local_enrichment_effective",
                "top1pct_in_local_roi_effective",
                "top1pct_local_enrichment_effective",
                "no_change_rate",
            ],
        ),
        (
            concept_rows,
            "factorized_concept_summary.csv",
            ["experiment_name", "group"],
            ["macro_ap", "macro_f1", "macro_balanced_accuracy"],
        ),
        (
            semantic_rows,
            "factorized_semantic_bottleneck_summary.csv",
            ["experiment_name", "condition"],
            ["mean_mse", "mean_l1", "mean_ssim", "mean_psnr", "mean_edge"],
        ),
        (
            probe_rows,
            "factorized_probe_summary.csv",
            ["experiment_name", "group"],
            ["macro_probe_ap", "macro_probe_ap_lift", "macro_probe_balanced_accuracy"],
        ),
    ]
    for rows, filename, group_columns, metrics in specifications:
        if rows:
            summarize(pd.DataFrame(rows), group_columns, metrics).to_csv(root / filename, index=False)


if __name__ == "__main__":
    main()
