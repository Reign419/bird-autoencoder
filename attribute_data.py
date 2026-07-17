"""CUB-200-2011 image-level attribute parsing and deterministic splits.

The 3.6M-row ``image_attribute_labels.txt`` file is intentionally parsed as a
stream and cached as compact NumPy arrays.  Attribute selection only uses the
official training partition; the official test partition is never consulted.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


N_CUB_IMAGES = 11_788
N_CUB_ATTRIBUTES = 312
CERTAINTY_WEIGHTS = np.asarray([0.0, 0.0, 0.25, 0.5, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class CUBAttributeData:
    image_table: pd.DataFrame
    attribute_table: pd.DataFrame
    labels: np.ndarray
    certainty: np.ndarray
    weights: np.ndarray


def _resolve_metadata_file(cub_root: Path, relative: str) -> Path:
    candidates = [cub_root / relative]
    if relative.startswith("attributes/"):
        candidates.append(cub_root / Path(relative).name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing CUB metadata file {relative!r}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def load_attribute_definitions(path: os.PathLike[str] | str) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                attribute_id_text, definition = raw.split(maxsplit=1)
                group, value = definition.split("::", maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"Malformed attribute definition on line {line_number}") from exc
            records.append(
                {
                    "attribute_id": int(attribute_id_text),
                    "definition": definition,
                    "group": group,
                    "value": value,
                }
            )
    table = pd.DataFrame.from_records(records)
    expected = np.arange(1, len(table) + 1)
    if table.empty or not np.array_equal(table["attribute_id"].to_numpy(), expected):
        raise ValueError("Attribute IDs must be contiguous and one-indexed")
    return table


def load_image_table(cub_root: os.PathLike[str] | str) -> pd.DataFrame:
    cub_root = Path(cub_root)
    images_path = _resolve_metadata_file(cub_root, "images.txt")
    split_path = _resolve_metadata_file(cub_root, "train_test_split.txt")

    images = pd.read_csv(images_path, sep=r"\s+", names=["image_id", "relative_path"])
    split = pd.read_csv(split_path, sep=r"\s+", names=["image_id", "is_train"])
    if len(images) != len(split) or not np.array_equal(images.image_id, split.image_id):
        raise ValueError("images.txt and train_test_split.txt image IDs are not aligned")
    if not np.array_equal(images.image_id.to_numpy(), np.arange(1, len(images) + 1)):
        raise ValueError("Image IDs must be contiguous and one-indexed")

    table = images.merge(split, on="image_id", validate="one_to_one")
    table["class_name"] = table.relative_path.str.split("/", n=1).str[0]
    table["class_id"] = table.class_name.str.split(".", n=1).str[0].astype(int)
    table["image_path"] = table.relative_path.map(lambda value: str(cub_root / "images" / value))
    table["official_split"] = np.where(table.is_train.eq(1), "official_train", "official_test")
    return table


def parse_attribute_labels(
    labels_path: os.PathLike[str] | str,
    n_images: int = N_CUB_IMAGES,
    n_attributes: int = N_CUB_ATTRIBUTES,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.zeros((n_images, n_attributes), dtype=np.uint8)
    certainty = np.zeros((n_images, n_attributes), dtype=np.uint8)
    seen = np.zeros((n_images, n_attributes), dtype=np.bool_)
    expected_rows = n_images * n_attributes
    row_count = 0
    repaired_rows = []

    with open(labels_path, "r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            fields = raw.split()
            # In the supplied CUB file, 606 records encode the timing field as
            # two whitespace-separated tokens, e.g. ``0  1.509``.  The first
            # four supervised fields remain intact.  Accept only this observed
            # six-column pattern; reject every other schema deviation.
            split_time_defect = len(fields) == 6 and fields[4] == "0"
            if len(fields) != 5 and not split_time_defect:
                raise ValueError(
                    f"Expected exactly five columns in image_attribute_labels.txt line "
                    f"{line_number} (or six with a zero time-prefix), "
                    f"found {len(fields)}"
                )
            if split_time_defect:
                repaired_rows.append((line_number, raw.rstrip("\n")))
                annotation_time_text = fields[5]
            else:
                annotation_time_text = fields[4]
            try:
                annotation_time = float(annotation_time_text)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid annotation time on line {line_number}: "
                    f"{annotation_time_text!r}"
                ) from exc
            if not np.isfinite(annotation_time) or annotation_time < 0:
                raise ValueError(
                    f"Invalid annotation time on line {line_number}: {annotation_time}"
                )
            image_id, attribute_id, is_present, certainty_id = map(int, fields[:4])
            image_index = image_id - 1
            attribute_index = attribute_id - 1
            if not 0 <= image_index < n_images:
                raise ValueError(f"Image ID out of range on line {line_number}: {image_id}")
            if not 0 <= attribute_index < n_attributes:
                raise ValueError(f"Attribute ID out of range on line {line_number}: {attribute_id}")
            if is_present not in (0, 1):
                raise ValueError(f"is_present must be 0 or 1 on line {line_number}")
            if certainty_id not in (1, 2, 3, 4):
                raise ValueError(f"certainty_id must be 1..4 on line {line_number}")
            if seen[image_index, attribute_index]:
                raise ValueError(
                    f"Duplicate image/attribute pair ({image_id}, {attribute_id}) on line {line_number}"
                )
            seen[image_index, attribute_index] = True
            labels[image_index, attribute_index] = is_present
            certainty[image_index, attribute_index] = certainty_id
            row_count += 1

    if row_count != expected_rows or not seen.all():
        missing = int(expected_rows - seen.sum())
        raise ValueError(
            f"Incomplete attribute file: expected {expected_rows:,} unique rows, "
            f"read {row_count:,}; missing {missing:,} pairs"
        )
    if repaired_rows:
        examples = "; ".join(
            f"line {line_number}: {content!r}"
            for line_number, content in repaired_rows[:3]
        )
        print(
            f"Warning: normalized {len(repaired_rows)} CUB split-time records; "
            f"the four supervised fields were unchanged. Examples: {examples}"
        )
    return labels, certainty


def load_or_create_attribute_cache(
    cub_root: os.PathLike[str] | str,
    cache_path: os.PathLike[str] | str | None = None,
) -> CUBAttributeData:
    cub_root = Path(cub_root)
    image_table = load_image_table(cub_root)
    attribute_path = _resolve_metadata_file(cub_root, "attributes/attributes.txt")
    attribute_table = load_attribute_definitions(attribute_path)
    if len(attribute_table) != N_CUB_ATTRIBUTES:
        raise ValueError(f"Expected {N_CUB_ATTRIBUTES} attributes, found {len(attribute_table)}")

    cache_path = Path(cache_path or cub_root / "cache" / "attribute_labels.npz")
    if cache_path.exists():
        cached = np.load(cache_path)
        labels = cached["labels"]
        certainty = cached["certainty"]
        if labels.shape != (len(image_table), len(attribute_table)):
            raise ValueError(f"Attribute cache has unexpected shape {labels.shape}")
    else:
        labels_path = _resolve_metadata_file(cub_root, "attributes/image_attribute_labels.txt")
        labels, certainty = parse_attribute_labels(
            labels_path,
            n_images=len(image_table),
            n_attributes=len(attribute_table),
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, labels=labels, certainty=certainty)

    weights = CERTAINTY_WEIGHTS[certainty]
    return CUBAttributeData(image_table, attribute_table, labels, certainty, weights)


def make_official_indices(
    image_table: pd.DataFrame,
    validation_fraction: float = 0.15,
    split_seed: int = 42,
) -> dict[str, np.ndarray]:
    official_train = image_table.index[image_table.is_train.eq(1)].to_numpy()
    official_test = image_table.index[image_table.is_train.eq(0)].to_numpy()
    class_ids = image_table.loc[official_train, "class_id"].to_numpy()
    train_indices, validation_indices = train_test_split(
        official_train,
        test_size=validation_fraction,
        random_state=split_seed,
        shuffle=True,
        stratify=class_ids,
    )
    return {
        "train": np.sort(train_indices),
        "validation": np.sort(validation_indices),
        "official_test": np.sort(official_test),
    }


def build_split_manifest(image_table: pd.DataFrame, splits: dict[str, np.ndarray]) -> pd.DataFrame:
    manifest = image_table.copy()
    manifest["experiment_split"] = ""
    manifest["split_index"] = -1
    for split_name, indices in splits.items():
        manifest.loc[indices, "experiment_split"] = split_name
        manifest.loc[indices, "split_index"] = np.arange(len(indices))
    return manifest


def compute_attribute_statistics(
    data: CUBAttributeData,
    indices: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    indices = np.asarray(list(indices), dtype=int)
    labels = data.labels[indices]
    weights = data.weights[indices]
    visible = weights > 0
    reliable = data.certainty[indices] >= 3
    effective_positive = np.sum(weights * labels, axis=0)
    effective_negative = np.sum(weights * (1 - labels), axis=0)
    denom = effective_positive + effective_negative

    atomic = data.attribute_table.copy()
    atomic["positive_count"] = labels.sum(axis=0)
    atomic["visible_count"] = visible.sum(axis=0)
    atomic["effective_positive"] = effective_positive
    atomic["effective_negative"] = effective_negative
    atomic["weighted_prevalence"] = np.divide(
        effective_positive,
        denom,
        out=np.zeros_like(effective_positive),
        where=denom > 0,
    )
    atomic["visibility"] = visible.mean(axis=0)
    atomic["reliable_fraction"] = reliable.mean(axis=0)

    rows = []
    for group, group_table in atomic.groupby("group", sort=False):
        columns = group_table.attribute_id.to_numpy() - 1
        group_visible_per_image = np.any(visible[:, columns], axis=1)
        group_reliable_per_image = np.any(reliable[:, columns], axis=1)
        rows.append(
            {
                "group": group,
                "n_attributes": len(columns),
                "group_visibility": float(group_visible_per_image.mean()),
                "group_reliable_fraction": float(group_reliable_per_image.mean()),
                "min_effective_positive": float(effective_positive[columns].min()),
                "median_effective_positive": float(np.median(effective_positive[columns])),
                "min_effective_negative": float(effective_negative[columns].min()),
                "eligible_attribute_fraction": float(
                    np.mean((effective_positive[columns] >= 50) & (effective_negative[columns] >= 50))
                ),
            }
        )
    return atomic, pd.DataFrame(rows)


def select_attribute_groups(
    atomic_statistics: pd.DataFrame,
    group_statistics: pd.DataFrame,
    min_group_visibility: float = 0.30,
    min_group_reliable_fraction: float = 0.30,
    min_eligible_attribute_fraction: float = 0.50,
    include_groups: Iterable[str] | None = None,
    exclude_groups: Iterable[str] | None = None,
) -> dict:
    include_groups = set(include_groups or [])
    exclude_groups = set(exclude_groups or [])
    selected, excluded = [], {}
    group_lookup = group_statistics.set_index("group")

    for group, row in group_lookup.iterrows():
        reasons = []
        if row.group_visibility < min_group_visibility:
            reasons.append("low_visibility")
        if row.group_reliable_fraction < min_group_reliable_fraction:
            reasons.append("low_reliability")
        if row.eligible_attribute_fraction < min_eligible_attribute_fraction:
            reasons.append("insufficient_balanced_support")
        if group in exclude_groups:
            reasons.append("explicitly_excluded")
        if group in include_groups:
            reasons = []
        if reasons:
            excluded[group] = reasons
        else:
            selected.append(group)

    selected_ids = atomic_statistics.loc[
        atomic_statistics.group.isin(selected), "attribute_id"
    ].astype(int).tolist()
    return {
        "selection_unit": "whole_attribute_group",
        "selected_groups": selected,
        "excluded_groups": excluded,
        "atomic_attribute_ids": selected_ids,
        "concept_dim": len(selected_ids),
        "thresholds": {
            "min_group_visibility": min_group_visibility,
            "min_group_reliable_fraction": min_group_reliable_fraction,
            "min_eligible_attribute_fraction": min_eligible_attribute_fraction,
        },
    }


def empirical_binary_entropy(labels: np.ndarray, weights: np.ndarray) -> float:
    positive = np.sum(labels * weights, axis=0)
    total = np.sum(weights, axis=0)
    probability = np.divide(positive, total, out=np.zeros_like(positive), where=total > 0)
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    entropy = -(probability * np.log2(probability) + (1 - probability) * np.log2(1 - probability))
    return float(entropy.sum())


def gaussian_noise_for_rate(concept_entropy_bits: float, dimensions: int) -> float:
    """AWGN proxy: solve D/2 log2(1 + 1/sigma^2) = concept entropy."""
    if dimensions <= 0 or concept_entropy_bits <= 0:
        return 1.0
    bits_per_dimension = concept_entropy_bits / dimensions
    denominator = math.pow(2.0, 2.0 * bits_per_dimension) - 1.0
    return float(math.sqrt(1.0 / max(denominator, 1e-8)))


def load_images(
    image_table: pd.DataFrame,
    indices: Iterable[int],
    img_size: tuple[int, int] = (64, 64),
) -> np.ndarray:
    images = []
    for index in indices:
        image_path = image_table.loc[int(index), "image_path"]
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((img_size[1], img_size[0]))
            images.append(np.asarray(image, dtype=np.float32) / 255.0)
    return np.asarray(images, dtype=np.float32)


def save_json(value: dict, path: os.PathLike[str] | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)