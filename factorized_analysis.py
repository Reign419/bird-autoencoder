"""Evaluation routines for factorized concept experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - metadata/probe-only environments
    tf = None

GROUP_PART_KEYWORDS = {
    "bill": ["beak"],
    "wing": ["left wing", "right wing"],
    "breast": ["breast"],
    "belly": ["belly"],
    "underparts": ["breast", "belly"],
    "back": ["back"],
    "upperparts": ["back", "left wing", "right wing"],
    "tail": ["tail"],
    "crown": ["crown"],
    "forehead": ["forehead"],
    "eye": ["left eye", "right eye"],
    "head": ["crown", "forehead", "left eye", "right eye", "beak"],
    "nape": ["nape"],
    "throat": ["throat"],
    "leg": ["left leg", "right leg"],
}


def build_part_rois(cub_root, image_ids, groups, output_shape=(64, 64), radius=6):
    """Build landmark-centred ROIs; CUB parts are points, not segmentations."""
    cub_root = Path(cub_root)
    names_path = cub_root / "parts" / "parts.txt"
    locations_path = cub_root / "parts" / "part_locs.txt"
    if not names_path.exists() or not locations_path.exists():
        return {}
    # part names contain spaces, so parse the first integer separately.
    name_records = []
    for raw in names_path.read_text(encoding="utf-8").splitlines():
        part_id, part_name = raw.split(maxsplit=1)
        name_records.append((int(part_id), part_name.lower()))
    part_names = dict(name_records)
    locations = pd.read_csv(
        locations_path,
        sep=r"\s+",
        names=["image_id", "part_id", "x", "y", "visible"],
    )
    locations = locations[locations.image_id.isin(image_ids) & locations.visible.eq(1)].copy()
    locations["part_name"] = locations.part_id.map(part_names)
    image_position = {int(image_id): position for position, image_id in enumerate(image_ids)}
    height, width = output_shape
    yy, xx = np.ogrid[:height, :width]
    images_table = pd.read_csv(
        cub_root / "images.txt", sep=r"\s+", names=["image_id", "relative_path"]
    ).set_index("image_id")
    original_sizes = {}
    for image_id in image_ids:
        path = cub_root / "images" / images_table.loc[int(image_id), "relative_path"]
        with Image.open(path) as image:
            original_sizes[int(image_id)] = image.size
    rois = {}
    for group in groups:
        keywords = []
        lowered = group.lower()
        for token, values in GROUP_PART_KEYWORDS.items():
            if token in lowered:
                keywords.extend(values)
        if not keywords:
            continue
        masks = np.zeros((len(image_ids), height, width), dtype=bool)
        subset = locations[locations.part_name.isin(set(keywords))]
        for row in subset.itertuples(index=False):
            position = image_position[int(row.image_id)]
            original_width, original_height = original_sizes[int(row.image_id)]
            center_x = float(row.x) * width / original_width
            center_y = float(row.y) * height / original_height
            masks[position] |= (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
        if masks.any():
            rois[group] = masks
    return rois


def build_bird_bboxes(cub_root, image_ids, output_shape=(64, 64)):
    """Build resized CUB bounding-box masks.

    Bounding boxes include some background and are not segmentation masks.  The
    function is evaluation-only; training continues to use uncropped images.
    """
    cub_root = Path(cub_root)
    boxes_path = cub_root / "bounding_boxes.txt"
    images_path = cub_root / "images.txt"
    if not boxes_path.exists() or not images_path.exists():
        return None
    boxes = pd.read_csv(
        boxes_path,
        sep=r"\s+",
        names=["image_id", "x", "y", "width", "height"],
    ).set_index("image_id")
    images_table = pd.read_csv(
        images_path,
        sep=r"\s+",
        names=["image_id", "relative_path"],
    ).set_index("image_id")
    output_height, output_width = output_shape
    masks = np.zeros((len(image_ids), output_height, output_width), dtype=bool)
    for position, image_id in enumerate(image_ids):
        image_id = int(image_id)
        if image_id not in boxes.index or image_id not in images_table.index:
            continue
        path = cub_root / "images" / images_table.loc[image_id, "relative_path"]
        with Image.open(path) as image:
            original_width, original_height = image.size
        box = boxes.loc[image_id]
        # CUB box origins are one-based.  Clip after resizing so malformed edge
        # cases cannot create negative NumPy slices.
        x0 = int(np.floor((float(box.x) - 1.0) * output_width / original_width))
        y0 = int(np.floor((float(box.y) - 1.0) * output_height / original_height))
        x1 = int(
            np.ceil((float(box.x) - 1.0 + float(box.width)) * output_width / original_width)
        )
        y1 = int(
            np.ceil((float(box.y) - 1.0 + float(box.height)) * output_height / original_height)
        )
        x0, x1 = np.clip([x0, x1], 0, output_width)
        y0, y1 = np.clip([y0, y1], 0, output_height)
        if x1 > x0 and y1 > y0:
            masks[position, y0:y1, x0:x1] = True
    return masks if masks.any() else None


def evaluate_concepts(labels, weights, probabilities, selected_attributes):
    rows = []
    for column, attribute in selected_attributes.reset_index(drop=True).iterrows():
        mask = weights[:, column] > 0
        y_true = labels[mask, column]
        y_score = probabilities[mask, column]
        if len(y_true) == 0 or np.unique(y_true).size < 2:
            ap = f1 = balanced = np.nan
        else:
            prediction = (y_score >= 0.5).astype(np.uint8)
            ap = average_precision_score(y_true, y_score)
            f1 = f1_score(y_true, prediction, zero_division=0)
            balanced = balanced_accuracy_score(y_true, prediction)
        rows.append(
            {
                "attribute_id": int(attribute.attribute_id),
                "group": attribute.group,
                "value": attribute.value,
                "visible_count": int(mask.sum()),
                "positive_count": int(y_true.sum()) if len(y_true) else 0,
                "average_precision": ap,
                "f1": f1,
                "balanced_accuracy": balanced,
            }
        )
    atomic = pd.DataFrame(rows)
    groups = (
        atomic.groupby("group", sort=False)
        .agg(
            n_attributes=("attribute_id", "size"),
            macro_ap=("average_precision", "mean"),
            macro_f1=("f1", "mean"),
            macro_balanced_accuracy=("balanced_accuracy", "mean"),
        )
        .reset_index()
    )
    return atomic, groups


def reconstruction_summary(name, images, predictions):
    from evaluate import per_image_reconstruction_metrics

    metrics = per_image_reconstruction_metrics(images, predictions)
    return {
        "condition": name,
        **{f"mean_{column}": float(metrics[column].mean()) for column in metrics.columns},
    }


def save_difference_montage(
    images,
    clean,
    counterfactual,
    path,
    title,
    count=6,
    difference_vmax=0.1,
):
    count = min(count, len(images))
    if count <= 0:
        return
    figure, axes = plt.subplots(count, 4, figsize=(10, 2.4 * count), squeeze=False)
    for row in range(count):
        difference = np.mean(np.abs(clean[row] - counterfactual[row]), axis=-1)
        axes[row, 0].imshow(images[row])
        axes[row, 1].imshow(clean[row])
        axes[row, 2].imshow(counterfactual[row])
        axes[row, 3].imshow(
            difference,
            cmap="magma",
            vmin=0,
            vmax=float(difference_vmax),
        )
        for axis in axes[row]:
            axis.axis("off")
    axes[0, 0].set_title("Input")
    axes[0, 1].set_title("Clean")
    axes[0, 2].set_title("Group shuffle")
    axes[0, 3].set_title("|Difference|")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def region_change_metrics(pixel_map, region, selector, top_fraction=0.01):
    """Return per-image target/non-target and top-change localization metrics."""
    pixel_map = np.asarray(pixel_map, dtype=np.float64)
    region = np.asarray(region, dtype=bool)
    selector = np.asarray(selector, dtype=bool)
    flat_map = pixel_map.reshape(len(pixel_map), -1)
    flat_region = region.reshape(len(region), -1)
    inside_area = flat_region.sum(axis=1)
    outside_area = (~flat_region).sum(axis=1)
    valid = selector & (inside_area > 0) & (outside_area > 0)
    if not valid.any():
        return None
    inside = np.sum(flat_map * flat_region, axis=1) / np.maximum(inside_area, 1)
    outside = np.sum(flat_map * (~flat_region), axis=1) / np.maximum(outside_area, 1)
    total = np.maximum(flat_map.sum(axis=1), 1e-8)
    inside_energy_fraction = np.sum(flat_map * flat_region, axis=1) / total
    area_fraction = inside_area / flat_region.shape[1]
    top_count = max(1, int(np.ceil(float(top_fraction) * flat_region.shape[1])))
    top_inside_fraction = np.zeros(len(pixel_map), dtype=np.float64)
    for index in np.flatnonzero(valid):
        top_indices = np.argpartition(flat_map[index], -top_count)[-top_count:]
        top_inside_fraction[index] = flat_region[index, top_indices].mean()
    return {
        "valid": valid,
        "inside": inside,
        "outside": outside,
        "enrichment": inside / np.maximum(outside, 1e-8),
        "inside_energy_fraction": inside_energy_fraction,
        "area_fraction": area_fraction,
        "top_inside_fraction": top_inside_fraction,
        "top_enrichment": top_inside_fraction / np.maximum(area_fraction, 1e-8),
    }


def evaluate_group_interventions(
    decoder,
    images,
    clean_reconstruction,
    residual,
    hard_concepts,
    selected_attributes,
    output_directory,
    batch_size=32,
    seed=42,
    concept_only=False,
    part_rois=None,
    bird_bboxes=None,
    difference_vmax=0.1,
    top_fraction=0.01,
):
    """Shuffle one complete group block at a time, preserving its marginal."""
    if tf is None:
        raise ImportError("TensorFlow is required for group intervention evaluation")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    selected_attributes = selected_attributes.reset_index(drop=True)

    for group, group_table in selected_attributes.groupby("group", sort=False):
        columns = group_table.index.to_numpy()
        permutation = rng.permutation(len(hard_concepts))
        counterfactual_concepts = hard_concepts.copy()
        counterfactual_concepts[:, columns] = hard_concepts[permutation][:, columns]
        effective = np.any(
            counterfactual_concepts[:, columns] != hard_concepts[:, columns],
            axis=1,
        )
        decoder_input = counterfactual_concepts if concept_only else [residual, counterfactual_concepts]
        counterfactual = decoder.predict(decoder_input, batch_size=batch_size, verbose=0)
        clean_ssim = tf.image.ssim(images, clean_reconstruction, max_val=1.0).numpy()
        shuffled_ssim = tf.image.ssim(images, counterfactual, max_val=1.0).numpy()
        pixel_map = np.mean(np.abs(clean_reconstruction - counterfactual), axis=-1)
        pixel_change = np.mean(pixel_map, axis=(1, 2))
        effect_mask = effective if effective.any() else np.ones_like(effective, dtype=bool)
        effective_flat = pixel_map[effect_mask].reshape(int(effect_mask.sum()), -1)
        row = {
                "group": group,
                "n_atomic_attributes": len(columns),
                "attempted": len(effective),
                "effective_changes": int(effective.sum()),
                "no_change_rate": float(1.0 - effective.mean()),
                "u_global_ssim_all": float(np.mean(clean_ssim - shuffled_ssim)),
                "u_global_ssim_effective": float(np.mean((clean_ssim - shuffled_ssim)[effect_mask])),
                "mean_pixel_change_all": float(pixel_change.mean()),
                "mean_pixel_change_effective": float(pixel_change[effect_mask].mean()),
                "pixel_change_p95_effective": float(
                    np.mean(np.quantile(effective_flat, 0.95, axis=1))
                ),
                "pixel_change_p99_effective": float(
                    np.mean(np.quantile(effective_flat, 0.99, axis=1))
                ),
                "note": "Global SSIM can underestimate localized concept effects.",
            }
        if bird_bboxes is not None:
            bbox_metrics = region_change_metrics(
                pixel_map,
                bird_bboxes,
                effect_mask,
                top_fraction=top_fraction,
            )
            if bbox_metrics is not None:
                valid = bbox_metrics["valid"]
                row.update(
                    {
                        "bird_bbox_pixel_change_effective": float(
                            np.mean(bbox_metrics["inside"][valid])
                        ),
                        "bird_outside_bbox_pixel_change_effective": float(
                            np.mean(bbox_metrics["outside"][valid])
                        ),
                        "bird_bbox_enrichment_effective": float(
                            np.mean(bbox_metrics["enrichment"][valid])
                        ),
                        "top1pct_in_bird_bbox_effective": float(
                            np.mean(bbox_metrics["top_inside_fraction"][valid])
                        ),
                        "top1pct_bird_bbox_enrichment_effective": float(
                            np.mean(bbox_metrics["top_enrichment"][valid])
                        ),
                        "bird_bbox_area_fraction": float(
                            np.mean(bbox_metrics["area_fraction"][valid])
                        ),
                        "bird_bbox_samples": int(valid.sum()),
                        "bird_bbox_note": "CUB bounding box, not a segmentation mask.",
                    }
                )
        if part_rois and group in part_rois:
            roi = part_rois[group]
            roi_metrics = region_change_metrics(
                pixel_map,
                roi,
                effective,
                top_fraction=top_fraction,
            )
            if roi_metrics is not None:
                valid = roi_metrics["valid"]
                # Preserve the two historical columns exactly.
                row["u_local_pixel_effective"] = float(
                    np.mean(roi_metrics["inside"][valid])
                )
                row["localization_ratio_effective"] = float(
                    np.mean(roi_metrics["inside_energy_fraction"][valid])
                )
                row["local_roi_samples"] = int(valid.sum())
                row["local_roi_note"] = "Landmark-centred ROI, not a segmentation mask."
                row["local_non_target_pixel_effective"] = float(
                    np.mean(roi_metrics["outside"][valid])
                )
                row["local_enrichment_effective"] = float(
                    np.mean(roi_metrics["enrichment"][valid])
                )
                row["top1pct_in_local_roi_effective"] = float(
                    np.mean(roi_metrics["top_inside_fraction"][valid])
                )
                row["top1pct_local_enrichment_effective"] = float(
                    np.mean(roi_metrics["top_enrichment"][valid])
                )
                row["local_roi_area_fraction"] = float(
                    np.mean(roi_metrics["area_fraction"][valid])
                )
        rows.append(row)
        safe_name = group.replace("has_", "").replace("/", "_")
        save_difference_montage(
            images[effect_mask],
            clean_reconstruction[effect_mask],
            counterfactual[effect_mask],
            output_directory / f"{safe_name}.png",
            title=f"Intervention: {group}",
            difference_vmax=difference_vmax,
        )
    return pd.DataFrame(rows)
