"""Freeze the predictability-filtered group set after a concept-only pilot."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-selection", required=True)
    parser.add_argument("--concept-metrics", required=True)
    parser.add_argument("--attribute-definitions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-group-ap-lift", type=float, default=0.05)
    args = parser.parse_args()

    selection = json.loads(Path(args.initial_selection).read_text(encoding="utf-8"))
    metrics = pd.read_csv(args.concept_metrics)
    metrics["prevalence_baseline_ap"] = metrics.positive_count / metrics.visible_count.clip(lower=1)
    metrics["ap_lift"] = metrics.average_precision - metrics.prevalence_baseline_ap
    group_metrics = (
        metrics.groupby("group", sort=False)
        .agg(macro_ap=("average_precision", "mean"), macro_ap_lift=("ap_lift", "mean"))
        .reset_index()
    )
    keep = set(
        group_metrics.loc[
            group_metrics.macro_ap_lift >= args.min_group_ap_lift, "group"
        ]
    )
    removed = [group for group in selection["selected_groups"] if group not in keep]
    definitions = pd.read_csv(args.attribute_definitions)
    selection["selected_groups"] = [
        group for group in selection["selected_groups"] if group in keep
    ]
    selection["atomic_attribute_ids"] = definitions.loc[
        definitions.group.isin(selection["selected_groups"]), "attribute_id"
    ].astype(int).tolist()
    selection["concept_dim"] = len(selection["atomic_attribute_ids"])
    selection["predictability_filter"] = {
        "min_group_macro_ap_lift": args.min_group_ap_lift,
        "removed_groups": removed,
        "source": str(args.concept_metrics),
        "data_scope": "official_train_validation_only",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    group_metrics.to_csv(output.with_suffix(".group_metrics.csv"), index=False)
    print(f"Retained {len(selection['selected_groups'])} groups / {selection['concept_dim']} atomic dimensions")


if __name__ == "__main__":
    main()
