"""Validate CUB attributes, create the cache, and freeze a group selection."""

import argparse
import json
from pathlib import Path

from attribute_data import (
    build_split_manifest,
    compute_attribute_statistics,
    empirical_binary_entropy,
    gaussian_noise_for_rate,
    load_or_create_attribute_cache,
    make_official_indices,
    save_json,
    select_attribute_groups,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cub-root", required=True)
    parser.add_argument("--output", default="outputs/attribute_preparation")
    parser.add_argument("--cache", default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--min-group-visibility", type=float, default=0.30)
    parser.add_argument("--min-group-reliability", type=float, default=0.30)
    parser.add_argument("--min-eligible-fraction", type=float, default=0.50)
    parser.add_argument("--include-group", action="append", default=[])
    parser.add_argument("--exclude-group", action="append", default=[])
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_or_create_attribute_cache(args.cub_root, args.cache)
    splits = make_official_indices(
        data.image_table,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
    )
    manifest = build_split_manifest(data.image_table, splits)
    atomic, groups = compute_attribute_statistics(data, splits["train"])
    selection = select_attribute_groups(
        atomic,
        groups,
        min_group_visibility=args.min_group_visibility,
        min_group_reliable_fraction=args.min_group_reliability,
        min_eligible_attribute_fraction=args.min_eligible_fraction,
        include_groups=args.include_group,
        exclude_groups=args.exclude_group,
    )
    selected_columns = [value - 1 for value in selection["atomic_attribute_ids"]]
    entropy = empirical_binary_entropy(
        data.labels[splits["train"]][:, selected_columns],
        data.weights[splits["train"]][:, selected_columns],
    )
    selection["empirical_concept_entropy_bits"] = entropy
    selection["recommended_u_noise_std_awgn_proxy"] = gaussian_noise_for_rate(
        entropy,
        selection["concept_dim"],
    )
    selection["split_seed"] = args.split_seed
    selection["selection_data"] = "official_train/train_subset_only"

    atomic.to_csv(output / "attribute_statistics.csv", index=False)
    groups.to_csv(output / "group_statistics.csv", index=False)
    manifest.to_csv(output / "split_manifest.csv", index=False)
    save_json(selection, output / "selected_attributes.json")

    report = [
        "# CUB attribute preparation",
        "",
        f"- Selected groups: {len(selection['selected_groups'])}",
        f"- Atomic concept dimension: {selection['concept_dim']}",
        f"- Empirical concept entropy: {entropy:.3f} bits",
        f"- AWGN rate-match proxy for u: {selection['recommended_u_noise_std_awgn_proxy']:.5f}",
        "",
        "## Selected groups",
        "",
        *[f"- {name}" for name in selection["selected_groups"]],
        "",
        "## Excluded groups",
        "",
        *[
            f"- {name}: {', '.join(reasons)}"
            for name, reasons in selection["excluded_groups"].items()
        ],
    ]
    (output / "attribute_selection_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
