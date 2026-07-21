"""Train a decoder-free CUB concept observability upper bound.

Architecture: shared convolutional trunk -> GAP -> Dense -> sigmoid. The model
has no residual branch, SemanticBottleneck, reconstruction decoder, or
reconstruction loss. Selection and reporting metrics use disjoint validation
subsets so the reported upper bound is not inflated by attribute selection.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model

from attribute_data import (
    build_split_manifest,
    load_images,
    load_or_create_attribute_cache,
    make_official_indices,
)
from factorized_analysis import evaluate_concepts
from losses import MaskedBinaryAccuracy, MaskedWeightedBinaryCrossentropy
from model.model_factorized_lite import build_factorized_lite_autoencoder
from run_factorized import validate_official_test_release
from train_utils import get_callbacks


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(value, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            default=lambda item: item.item() if isinstance(item, np.generic) else str(item),
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def pack_targets(labels, weights):
    return np.concatenate(
        [labels.astype(np.float32), weights.astype(np.float32)], axis=-1
    )


def split_selection_and_reporting(
    validation_indices,
    image_table,
    reporting_fraction=0.5,
    seed=1041,
):
    """Create disjoint, class-stratified selection and reporting subsets."""
    validation_indices = np.asarray(validation_indices, dtype=int)
    if not 0.0 < reporting_fraction < 1.0:
        raise ValueError("reporting_fraction must be between 0 and 1")
    class_ids = image_table.loc[validation_indices, "class_id"].to_numpy()
    selection, reporting = train_test_split(
        validation_indices,
        test_size=reporting_fraction,
        random_state=int(seed),
        shuffle=True,
        stratify=class_ids,
    )
    return np.sort(selection), np.sort(reporting)


def concept_tables(labels, weights, probabilities, selected_attributes):
    return evaluate_concepts(labels, weights, probabilities, selected_attributes)


def model_summary_text(model):
    lines = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines) + "\n"


def main(config_path):
    config = load_json(config_path)
    validate_official_test_release(config)
    seed = int(config.get("training_seed", 41))
    set_seed(seed)

    cub_root = Path(config["cub_root"])
    output = Path(config.get("output_path", "outputs/standalone_concept_pilot"))
    output.mkdir(parents=True, exist_ok=True)
    selection = load_json(config["selected_attributes_path"])
    selected_ids = [int(value) for value in selection["atomic_attribute_ids"]]
    if not selected_ids:
        raise ValueError("The selected attribute list is empty")
    selected_columns = np.asarray(selected_ids, dtype=int) - 1

    data = load_or_create_attribute_cache(cub_root, config.get("attribute_cache"))
    selected_attributes = (
        data.attribute_table.set_index("attribute_id").loc[selected_ids].reset_index()
    )
    base_splits = make_official_indices(
        data.image_table,
        validation_fraction=float(config.get("validation_fraction", 0.15)),
        split_seed=int(config.get("split_seed", 42)),
    )
    selection_validation, reporting_validation = split_selection_and_reporting(
        base_splits["validation"],
        data.image_table,
        reporting_fraction=float(
            config.get("reporting_fraction_of_validation", 0.5)
        ),
        seed=int(config.get("reporting_split_seed", 1041)),
    )
    splits = {
        "train": base_splits["train"],
        "selection_validation": selection_validation,
        "reporting_validation": reporting_validation,
        "official_test": base_splits["official_test"],
    }

    evaluate_official_test = bool(config.get("evaluate_official_test", False))
    active_splits = ["train", "selection_validation", "reporting_validation"]
    if evaluate_official_test:
        active_splits.append("official_test")

    img_size = tuple(config.get("img_size", [64, 64]))
    images = {
        name: load_images(data.image_table, splits[name], img_size=img_size)
        for name in active_splits
    }
    labels = {
        name: data.labels[splits[name]][:, selected_columns].astype(np.float32)
        for name in active_splits
    }
    weights = {
        name: data.weights[splits[name]][:, selected_columns].astype(np.float32)
        for name in active_splits
    }

    effective_positive = np.sum(labels["train"] * weights["train"], axis=0)
    effective_negative = np.sum((1.0 - labels["train"]) * weights["train"], axis=0)
    positive_weights = np.clip(
        effective_negative / np.maximum(effective_positive, 1e-6),
        1.0,
        float(config.get("positive_class_weight_cap", 10.0)),
    ).astype(np.float32)

    # Reuse the factorized shared trunk and concept head, then prune every
    # residual/bottleneck/decoder branch from the standalone graph.
    _, encoder, _ = build_factorized_lite_autoencoder(
        img_shape=(img_size[0], img_size[1], 3),
        concept_dim=len(selected_ids),
        residual_channels=15,
        max_residual_channels=15,
        condition_channels=int(config.get("condition_channels", 4)),
        latent_grid_size=int(config.get("latent_grid_size", 8)),
        base_channels=int(config.get("base_channels", 64)),
        max_channels=int(config.get("max_channels", 256)),
        mode="concept",
        semantic_method="ste",
        residual_dropout=0.0,
        residual_noise_std=0.0,
    )
    probabilities = encoder.get_layer("concepts").output
    predictor = Model(
        encoder.input, probabilities, name="standalone_concept_predictor"
    )
    predictor.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(config.get("learning_rate", 1e-3)),
            clipnorm=float(config.get("gradient_clipnorm", 1.0)),
        ),
        loss=MaskedWeightedBinaryCrossentropy(
            positive_weights=positive_weights.tolist()
        ),
        metrics=[MaskedBinaryAccuracy()],
    )

    callbacks = get_callbacks(
        str(output),
        monitor="val_loss",
        mode="min",
        early_stopping_patience=int(config.get("early_stopping_patience", 8)),
        reduce_lr_patience=int(config.get("reduce_lr_patience", 3)),
    )
    callbacks.append(tf.keras.callbacks.CSVLogger(output / "history.csv"))
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    history = predictor.fit(
        images["train"],
        pack_targets(labels["train"], weights["train"]),
        validation_data=(
            images["selection_validation"],
            pack_targets(
                labels["selection_validation"], weights["selection_validation"]
            ),
        ),
        epochs=int(config.get("epochs", 60)),
        batch_size=int(config.get("batch_size", 32)),
        callbacks=callbacks,
        verbose=1,
    )

    batch_size = int(config.get("batch_size", 32))
    selection_probabilities = predictor.predict(
        images["selection_validation"], batch_size=batch_size, verbose=0
    )
    selection_atomic, selection_groups = concept_tables(
        labels["selection_validation"],
        weights["selection_validation"],
        selection_probabilities,
        selected_attributes,
    )
    # Backward-compatible names are explicitly the selection tables used by
    # refine_attribute_selection.py.
    selection_atomic.to_csv(output / "concept_metrics.csv", index=False)
    selection_groups.to_csv(output / "concept_group_metrics.csv", index=False)

    reporting_probabilities = predictor.predict(
        images["reporting_validation"], batch_size=batch_size, verbose=0
    )
    reporting_atomic, reporting_groups = concept_tables(
        labels["reporting_validation"],
        weights["reporting_validation"],
        reporting_probabilities,
        selected_attributes,
    )
    reporting_atomic.to_csv(
        output / "reporting_concept_metrics.csv", index=False
    )
    reporting_groups.to_csv(
        output / "reporting_concept_group_metrics.csv", index=False
    )

    build_split_manifest(data.image_table, splits).to_csv(
        output / "split_manifest.csv", index=False
    )
    selected_attributes.to_csv(
        output / "selected_attribute_definitions.csv", index=False
    )
    (output / "model_summary.txt").write_text(
        model_summary_text(predictor), encoding="utf-8"
    )

    result = {
        "run_name": config.get("run_name", "pilot_standalone_concept"),
        "training_seed": seed,
        "split_seed": int(config.get("split_seed", 42)),
        "reporting_split_seed": int(config.get("reporting_split_seed", 1041)),
        "concept_dim": len(selected_ids),
        "architecture": "shared_trunk_gap_dense_sigmoid",
        "uses_semantic_bottleneck": False,
        "uses_residual": False,
        "uses_decoder": False,
        "parameter_count": predictor.count_params(),
        "train_size": len(splits["train"]),
        "selection_validation_size": len(splits["selection_validation"]),
        "reporting_validation_size": len(splits["reporting_validation"]),
        "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        "selection_macro_ap": float(selection_atomic.average_precision.mean()),
        "selection_macro_f1": float(selection_atomic.f1.mean()),
        "selection_macro_balanced_accuracy": float(
            selection_atomic.balanced_accuracy.mean()
        ),
        "reporting_macro_ap": float(reporting_atomic.average_precision.mean()),
        "reporting_macro_f1": float(reporting_atomic.f1.mean()),
        "reporting_macro_balanced_accuracy": float(
            reporting_atomic.balanced_accuracy.mean()
        ),
        "evaluate_official_test": evaluate_official_test,
    }

    if evaluate_official_test:
        test_probabilities = predictor.predict(
            images["official_test"], batch_size=batch_size, verbose=0
        )
        test_atomic, test_groups = concept_tables(
            labels["official_test"],
            weights["official_test"],
            test_probabilities,
            selected_attributes,
        )
        test_atomic.to_csv(
            output / "official_test_concept_metrics.csv", index=False
        )
        test_groups.to_csv(
            output / "official_test_concept_group_metrics.csv", index=False
        )
        result["official_test_macro_ap"] = float(
            test_atomic.average_precision.mean()
        )

    save_json(result, output / "result.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/standalone_concept_pilot.json"
    )
    arguments = parser.parse_args()
    main(arguments.config)
