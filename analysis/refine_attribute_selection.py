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
    parser.add_argument("--min-attribute-ap-lift", type=float, default=-1.0)
    parser.add_argument("--min-balanced-accuracy", type=float, default=0.0)
    parser.add_argument("--min-positive-count", type=int, default=0)
    parser.add_argument("--min-negative-count", type=int, default=0)
    parser.add_argument("--min-predictable-fraction", type=float, default=0.0)
    args = parser.parse_args()
    if not 0.0 <= args.min_predictable_fraction <= 1.0:
        parser.error("--min-predictable-fraction must be between 0 and 1")

    selection = json.loads(Path(args.initial_selection).read_text(encoding="utf-8"))
    metrics = pd.read_csv(args.concept_metrics)
    metrics["prevalence_baseline_ap"] = metrics.positive_count / metrics.visible_count.clip(lower=1)
    metrics["ap_lift"] = metrics.average_precision - metrics.prevalence_baseline_ap
    metrics["negative_count"] = metrics.visible_count - metrics.positive_count
    metrics["predictable_atomic"] = (
        metrics.ap_lift.ge(args.min_attribute_ap_lift)
        & metrics.balanced_accuracy.ge(args.min_balanced_accuracy)
        & metrics.positive_count.ge(args.min_positive_count)
        & metrics.negative_count.ge(args.min_negative_count)
    )
    group_metrics = (
        metrics.groupby("group", sort=False)
        .agg(
            n_attributes=("attribute_id", "size"),
            macro_ap=("average_precision", "mean"),
            macro_ap_lift=("ap_lift", "mean"),
            macro_balanced_accuracy=("balanced_accuracy", "mean"),
            predictable_attributes=("predictable_atomic", "sum"),
            predictable_fraction=("predictable_atomic", "mean"),
        )
        .reset_index()
    )
    group_metrics["selected_complete_group"] = (
        group_metrics.macro_ap_lift.ge(args.min_group_ap_lift)
        & group_metrics.predictable_fraction.ge(args.min_predictable_fraction)
    )
    keep = set(
        group_metrics.loc[
            group_metrics.selected_complete_group, "group"
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
        "min_attribute_ap_lift": args.min_attribute_ap_lift,
        "min_balanced_accuracy": args.min_balanced_accuracy,
        "min_positive_count": args.min_positive_count,
        "min_negative_count": args.min_negative_count,
        "min_predictable_fraction": args.min_predictable_fraction,
        "removed_groups": removed,
        "source": str(args.concept_metrics),
        "data_scope": "official_train_validation_only",
        "selection_unit": "complete_natural_group",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    group_metrics.to_csv(output.with_suffix(".group_metrics.csv"), index=False)
    metrics["selected_complete_group"] = metrics.group.isin(keep)
    metrics.to_csv(output.with_suffix(".attribute_metrics.csv"), index=False)
    group_metrics.loc[~group_metrics.selected_complete_group].to_csv(
        output.with_suffix(".excluded_groups.csv"), index=False
    )
    atomic_ids = metrics.loc[metrics.predictable_atomic, "attribute_id"].astype(int).tolist()
    atomic_selection = {
        "atomic_attribute_ids": atomic_ids,
        "concept_dim": len(atomic_ids),
        "source": str(args.concept_metrics),
        "data_scope": "official_train_validation_only",
        "selection_unit": "atomic_attribute_secondary_analysis",
        "thresholds": selection["predictability_filter"],
    }
    output.with_suffix(".atomic.json").write_text(
        json.dumps(atomic_selection, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Retained {len(selection['selected_groups'])} groups / {selection['concept_dim']} atomic dimensions")


if __name__ == "__main__":
    main()
