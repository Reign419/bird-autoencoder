"""Train CUB concept/residual factorized models with official splits."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from attribute_data import (
    build_split_manifest,
    empirical_binary_entropy,
    load_images,
    load_or_create_attribute_cache,
    make_official_indices,
)
from factorized_analysis import (
    build_bird_bboxes,
    build_part_rois,
    evaluate_concepts,
    evaluate_group_interventions,
    reconstruction_summary,
)
from losses import (
    MaskedBinaryAccuracy,
    MaskedWeightedBinaryCrossentropy,
    edge_metric,
    l1_metric,
    make_reconstruction_loss,
    mse_metric,
    psnr_metric,
    ssim_loss_metric,
    ssim_metric,
)
from model.model_factorized_lite import TemperatureAnnealing, build_factorized_lite_autoencoder
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


class MetricAliasCallback(tf.keras.callbacks.Callback):
    """Expose a stable monitor key across Keras multi-output naming variants."""

    def __init__(self, alias):
        super().__init__()
        self.alias = alias

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or self.alias in logs:
            return
        candidates = [
            key
            for key in logs
            if key.startswith("val_") and key.endswith("ssim_metric")
        ]
        if candidates:
            candidates.sort(key=lambda key: ("reconstruction" not in key, key))
            logs[self.alias] = logs[candidates[0]]


def resolve_history_monitor(history, requested):
    if requested in history:
        return requested
    if requested.endswith("ssim_metric"):
        candidates = [
            key
            for key in history
            if key.startswith("val_") and key.endswith("ssim_metric")
        ]
        if candidates:
            candidates.sort(key=lambda key: ("reconstruction" not in key, key))
            return candidates[0]
    raise KeyError(
        f"Configured monitor {requested!r} was not produced by Keras. "
        f"Available history keys: {', '.join(sorted(history))}"
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def limited_indices(indices, maximum, seed):
    if maximum is None or maximum <= 0 or len(indices) <= maximum:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=int(maximum), replace=False))


def pack_concept_targets(labels, weights):
    return np.concatenate(
        [labels.astype(np.float32), weights.astype(np.float32)], axis=-1
    )


def expand_experiments(config):
    seeds = config.get("training_seeds", [config.get("training_seed", 42)])
    expanded = []
    for experiment in config["experiments"]:
        experiment_seeds = (
            [experiment["training_seed"]]
            if "training_seed" in experiment
            else seeds
        )
        for seed in experiment_seeds:
            item = dict(experiment)
            item["training_seed"] = int(seed)
            expanded.append(item)
    return expanded


def build_callbacks(run_directory, experiment, config, model):
    monitor = experiment.get(
        "monitor", config.get("monitor", "val_reconstruction_ssim_metric")
    )
    if experiment.get("mode", "concept") == "control" and monitor.startswith(
        "val_reconstruction_"
    ):
        monitor = "val_" + monitor.removeprefix("val_reconstruction_")
    mode = experiment.get("monitor_mode", config.get("monitor_mode", "max"))
    callbacks = [MetricAliasCallback(monitor)]
    callbacks.extend(
        get_callbacks(
            str(run_directory),
            monitor=monitor,
            mode=mode,
            early_stopping_patience=int(config.get("early_stopping_patience", 10)),
            reduce_lr_patience=int(config.get("reduce_lr_patience", 3)),
        )
    )
    callbacks.append(tf.keras.callbacks.CSVLogger(run_directory / "history.csv"))
    callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    if experiment.get("semantic_method", "ste") == "gumbel":
        callbacks.append(
            TemperatureAnnealing(
                start=float(experiment.get("temperature_start", 1.0)),
                end=float(experiment.get("temperature_end", 0.2)),
                epochs=int(
                    experiment.get(
                        "temperature_anneal_epochs", config.get("epochs", 60)
                    )
                ),
            )
        )
    return callbacks, monitor, mode


def model_summary_text(model):
    lines = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines) + "\n"


def main(config_path):
    config = load_json(config_path)
    # This guard runs even when callers bypass run_factorized.py.
    validate_official_test_release(config)

    output_root = Path(config["output_path"])
    output_root.mkdir(parents=True, exist_ok=True)
    cub_root = Path(config["cub_root"])
    selection = load_json(config["selected_attributes_path"])
    selected_ids = [int(value) for value in selection["atomic_attribute_ids"]]
    selected_columns = np.asarray(selected_ids, dtype=int) - 1
    concept_dim = len(selected_columns)
    if concept_dim == 0:
        raise ValueError("The selected attribute list is empty")

    data = load_or_create_attribute_cache(cub_root, config.get("attribute_cache"))
    selected_attributes = (
        data.attribute_table.set_index("attribute_id").loc[selected_ids].reset_index()
    )
    split_seed = int(config.get("split_seed", 42))
    splits = make_official_indices(
        data.image_table,
        validation_fraction=float(config.get("validation_fraction", 0.15)),
        split_seed=split_seed,
    )
    maximum = config.get("max_samples_per_split")
    splits = {
        name: limited_indices(indices, maximum, split_seed + offset)
        for offset, (name, indices) in enumerate(splits.items())
    }
    manifest = build_split_manifest(data.image_table, splits)
    selected_entropy_bits = empirical_binary_entropy(
        data.labels[splits["train"]][:, selected_columns],
        data.weights[splits["train"]][:, selected_columns],
    )
    img_size = tuple(config.get("img_size", [64, 64]))
    evaluate_official_test = bool(config.get("evaluate_official_test", False))
    active_split_names = ["train", "validation"]
    if evaluate_official_test:
        active_split_names.append("official_test")
    arrays = {
        name: load_images(data.image_table, splits[name], img_size=img_size)
        for name in active_split_names
    }
    labels = {
        name: data.labels[indices][:, selected_columns].astype(np.float32)
        for name, indices in splits.items()
        if name in active_split_names
    }
    weights = {
        name: data.weights[indices][:, selected_columns].astype(np.float32)
        for name, indices in splits.items()
        if name in active_split_names
    }
    effective_positive = np.sum(labels["train"] * weights["train"], axis=0)
    effective_negative = np.sum((1.0 - labels["train"]) * weights["train"], axis=0)
    positive_weight_cap = float(config.get("positive_class_weight_cap", 10.0))
    positive_class_weights = np.clip(
        effective_negative / np.maximum(effective_positive, 1e-6),
        1.0,
        positive_weight_cap,
    ).astype(np.float32)
    manifest.to_csv(output_root / "split_manifest.csv", index=False)
    selected_attributes.to_csv(
        output_root / "selected_attribute_definitions.csv", index=False
    )
    save_json(selection, output_root / "selected_attributes.json")

    results = []
    for experiment in expand_experiments(config):
        tf.keras.backend.clear_session()
        seed = int(experiment["training_seed"])
        set_seed(seed)
        mode = experiment.get("mode", "concept")
        if mode not in {"concept", "control", "concept_only"}:
            raise ValueError(f"Unknown factorized mode {mode!r}")

        model, encoder, decoder = build_factorized_lite_autoencoder(
            img_shape=(img_size[0], img_size[1], 3),
            concept_dim=concept_dim,
            residual_channels=int(experiment.get("residual_channels", 15)),
            max_residual_channels=int(
                experiment.get(
                    "max_residual_channels", config.get("max_residual_channels", 15)
                )
            ),
            condition_channels=int(experiment.get("condition_channels", 4)),
            latent_grid_size=int(experiment.get("latent_grid_size", 8)),
            base_channels=int(experiment.get("base_channels", 64)),
            max_channels=int(experiment.get("max_channels", 256)),
            mode=mode,
            semantic_method=experiment.get("semantic_method", "ste"),
            semantic_temperature=float(experiment.get("temperature_start", 1.0)),
            residual_dropout=float(experiment.get("residual_dropout", 0.1)),
            residual_noise_std=float(experiment.get("residual_noise_std", 0.05)),
            control_noise_std=0.0,
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{experiment['name']}_Dc{concept_dim}_seed{seed}_{timestamp}"
        run_directory = output_root / run_name
        run_directory.mkdir(parents=True, exist_ok=True)
        (run_directory / "figures" / "group_interventions").mkdir(
            parents=True, exist_ok=True
        )

        reconstruction_loss = make_reconstruction_loss(
            **config.get("reconstruction_loss", {})
        )
        learning_rate = float(
            experiment.get("learning_rate", config.get("learning_rate", 1e-3))
        )
        optimizer = tf.keras.optimizers.Adam(
            learning_rate,
            clipnorm=float(config.get("gradient_clipnorm", 1.0)),
        )
        reconstruction_metrics = [
            mse_metric,
            l1_metric,
            ssim_metric,
            ssim_loss_metric,
            edge_metric,
            psnr_metric,
        ]
        if mode in {"concept", "concept_only"}:
            model.compile(
                optimizer=optimizer,
                loss={
                    "reconstruction": reconstruction_loss,
                    "concepts": MaskedWeightedBinaryCrossentropy(
                        positive_weights=positive_class_weights.tolist()
                    ),
                },
                loss_weights={
                    "reconstruction": float(
                        experiment.get("reconstruction_weight", 1.0)
                    ),
                    "concepts": float(experiment.get("concept_weight", 1.0)),
                },
                metrics={
                    "reconstruction": reconstruction_metrics,
                    "concepts": [MaskedBinaryAccuracy()],
                },
            )
            train_targets = {
                "reconstruction": arrays["train"],
                "concepts": pack_concept_targets(
                    labels["train"], weights["train"]
                ),
            }
            validation_targets = {
                "reconstruction": arrays["validation"],
                "concepts": pack_concept_targets(
                    labels["validation"], weights["validation"]
                ),
            }
        else:
            model.compile(
                optimizer=optimizer,
                loss=reconstruction_loss,
                metrics=reconstruction_metrics,
            )
            train_targets = arrays["train"]
            validation_targets = arrays["validation"]

        callbacks, monitor, monitor_mode = build_callbacks(
            run_directory, experiment, config, model
        )
        run_config = {
            **experiment,
            "cub_root": str(cub_root),
            "concept_dim": concept_dim,
            "selected_groups": selection["selected_groups"],
            "selected_attribute_ids": selected_ids,
            "split_seed": split_seed,
            "img_size": list(img_size),
            "train_size": len(arrays["train"]),
            "validation_size": len(arrays["validation"]),
            "official_test_size": (
                len(arrays["official_test"]) if "official_test" in arrays else 0
            ),
            "empirical_concept_entropy_bits": selected_entropy_bits,
            "positive_class_weight_cap": positive_weight_cap,
            "monitor": monitor,
            "monitor_mode": monitor_mode,
            "evaluate_official_test": evaluate_official_test,
            "official_test_release": bool(
                config.get("official_test_release", False)
            ),
            "control_topology": (
                "dense_sigmoid_semantic_bottleneck_no_concept_loss"
                if mode == "control"
                else None
            ),
        }
        save_json(run_config, run_directory / "config.json")
        (run_directory / "model_summary.txt").write_text(
            model_summary_text(model), encoding="utf-8"
        )
        (run_directory / "encoder_summary.txt").write_text(
            model_summary_text(encoder), encoding="utf-8"
        )
        (run_directory / "decoder_summary.txt").write_text(
            model_summary_text(decoder), encoding="utf-8"
        )

        history = model.fit(
            arrays["train"],
            train_targets,
            validation_data=(arrays["validation"], validation_targets),
            epochs=int(experiment.get("epochs", config.get("epochs", 60))),
            batch_size=int(
                experiment.get("batch_size", config.get("batch_size", 32))
            ),
            callbacks=callbacks,
            verbose=1,
        )
        batch_size = int(
            experiment.get("batch_size", config.get("batch_size", 32))
        )
        actual_monitor = resolve_history_monitor(history.history, monitor)
        monitor_values = history.history[actual_monitor]
        monitor = actual_monitor
        best_index = int(
            np.argmax(monitor_values)
            if monitor_mode == "max"
            else np.argmin(monitor_values)
        )
        result = {
            "run_name": run_name,
            "experiment_name": experiment["name"],
            "mode": mode,
            "training_seed": seed,
            "concept_dim": concept_dim,
            "active_residual_channels": int(
                experiment.get("residual_channels", 15)
            ),
            "max_residual_channels": int(
                experiment.get(
                    "max_residual_channels", config.get("max_residual_channels", 15)
                )
            ),
            "best_epoch": best_index + 1,
            "best_monitor_value": float(monitor_values[best_index]),
            "parameter_count": model.count_params(),
            "encoder_parameter_count": encoder.count_params(),
            "decoder_parameter_count": decoder.count_params(),
        }

        validation_prediction = model.predict(
            arrays["validation"], batch_size=batch_size, verbose=0
        )
        if mode in {"concept", "concept_only"}:
            validation_reconstruction = validation_prediction["reconstruction"]
            probabilities = validation_prediction["concepts"]
            encoded = encoder.predict(
                arrays["validation"], batch_size=batch_size, verbose=0
            )
            hard = (probabilities >= 0.5).astype(np.float32)
            residual = encoded.get("residual")
            clean_input = hard if mode == "concept_only" else [residual, hard]
            clean_reconstruction = decoder.predict(
                clean_input, batch_size=batch_size, verbose=0
            )
            run_reconstruction_analysis = (
                float(experiment.get("reconstruction_weight", 1.0)) > 0
                and bool(experiment.get("run_reconstruction_analysis", True))
            )
            if run_reconstruction_analysis:
                soft_input = (
                    probabilities
                    if mode == "concept_only"
                    else [residual, probabilities]
                )
                soft_reconstruction = decoder.predict(
                    soft_input, batch_size=batch_size, verbose=0
                )
                ground_truth = np.where(
                    weights["validation"] > 0, labels["validation"], hard
                )
                ground_truth_input = (
                    ground_truth
                    if mode == "concept_only"
                    else [residual, ground_truth]
                )
                ground_truth_reconstruction = decoder.predict(
                    ground_truth_input, batch_size=batch_size, verbose=0
                )
                semantic_rows = [
                    reconstruction_summary(
                        "model_hard", arrays["validation"], clean_reconstruction
                    ),
                    reconstruction_summary(
                        "soft_concepts", arrays["validation"], soft_reconstruction
                    ),
                    reconstruction_summary(
                        "ground_truth_visible",
                        arrays["validation"],
                        ground_truth_reconstruction,
                    ),
                ]
                pd.DataFrame(semantic_rows).to_csv(
                    run_directory / "semantic_bottleneck_analysis.csv", index=False
                )
            atomic_metrics, group_metrics = evaluate_concepts(
                labels["validation"],
                weights["validation"],
                probabilities,
                selected_attributes,
            )
            atomic_metrics.to_csv(
                run_directory / "concept_metrics.csv", index=False
            )
            group_metrics.to_csv(
                run_directory / "concept_group_metrics.csv", index=False
            )
            result["validation_macro_ap"] = float(
                atomic_metrics.average_precision.mean()
            )
            result["validation_macro_f1"] = float(atomic_metrics.f1.mean())

            if run_reconstruction_analysis:
                intervention_count = min(
                    int(config.get("intervention_max_images", 256)),
                    len(arrays["validation"]),
                )
                intervention_indices = np.arange(intervention_count)
                intervention_image_ids = (
                    splits["validation"][intervention_indices] + 1
                )
                part_rois = build_part_rois(
                    cub_root,
                    intervention_image_ids,
                    selected_attributes.group.unique(),
                    output_shape=img_size,
                    radius=int(config.get("part_roi_radius", 6)),
                )
                bird_bboxes = build_bird_bboxes(
                    cub_root,
                    intervention_image_ids,
                    output_shape=img_size,
                )
                intervention_table = evaluate_group_interventions(
                    decoder=decoder,
                    images=arrays["validation"][intervention_indices],
                    clean_reconstruction=clean_reconstruction[intervention_indices],
                    residual=(
                        None if residual is None else residual[intervention_indices]
                    ),
                    hard_concepts=hard[intervention_indices],
                    selected_attributes=selected_attributes,
                    output_directory=(
                        run_directory / "figures" / "group_interventions"
                    ),
                    batch_size=batch_size,
                    seed=seed,
                    concept_only=mode == "concept_only",
                    part_rois=part_rois,
                    bird_bboxes=bird_bboxes,
                    difference_vmax=float(
                        config.get("difference_map_vmax", 0.1)
                    ),
                    top_fraction=0.01,
                )
                intervention_table.to_csv(
                    run_directory / "group_interventions.csv", index=False
                )
                result["mean_group_u_global_ssim"] = float(
                    intervention_table.u_global_ssim_effective.mean()
                )
            np.savez_compressed(
                run_directory / "validation_latents.npz",
                residual=np.asarray([]) if residual is None else residual,
                concept_probabilities=probabilities,
                hard_concepts=hard,
                labels=labels["validation"],
                weights=weights["validation"],
                image_indices=splits["validation"],
                attribute_ids=np.asarray(selected_ids),
            )
            if mode == "concept" and bool(config.get("save_probe_latents", True)):
                train_encoded = encoder.predict(
                    arrays["train"], batch_size=batch_size, verbose=0
                )
                np.savez_compressed(
                    run_directory / "train_probe_latents.npz",
                    residual=train_encoded["residual"],
                    labels=labels["train"],
                    weights=weights["train"],
                    image_indices=splits["train"],
                    attribute_ids=np.asarray(selected_ids),
                )
        else:
            encoded = encoder.predict(
                arrays["validation"], batch_size=batch_size, verbose=0
            )
            validation_reconstruction = decoder.predict(
                [encoded["residual"], encoded["control"]],
                batch_size=batch_size,
                verbose=0,
            )
            pd.DataFrame(
                [
                    reconstruction_summary(
                        "matched_binary_control",
                        arrays["validation"],
                        validation_reconstruction,
                    )
                ]
            ).to_csv(
                run_directory / "control_bottleneck_analysis.csv", index=False
            )
            np.savez_compressed(
                run_directory / "validation_latents.npz",
                residual=encoded["residual"],
                control=encoded["control"],
                control_probabilities=encoded["control_probabilities"],
                image_indices=splits["validation"],
            )

        validation_summary = reconstruction_summary(
            "validation", arrays["validation"], validation_reconstruction
        )
        result.update(validation_summary)
        save_json(result, run_directory / "result.json")
        results.append(result)

        if evaluate_official_test:
            test_prediction = model.predict(
                arrays["official_test"], batch_size=batch_size, verbose=0
            )
            test_reconstruction = (
                test_prediction["reconstruction"]
                if isinstance(test_prediction, dict)
                else test_prediction
            )
            if mode == "control":
                test_encoded = encoder.predict(
                    arrays["official_test"], batch_size=batch_size, verbose=0
                )
                test_reconstruction = decoder.predict(
                    [test_encoded["residual"], test_encoded["control"]],
                    batch_size=batch_size,
                    verbose=0,
                )
            test_summary = reconstruction_summary(
                "official_test", arrays["official_test"], test_reconstruction
            )
            if mode in {"concept", "concept_only"}:
                test_probabilities = test_prediction["concepts"]
                test_atomic, test_groups = evaluate_concepts(
                    labels["official_test"],
                    weights["official_test"],
                    test_probabilities,
                    selected_attributes,
                )
                test_atomic.to_csv(
                    run_directory / "official_test_concept_metrics.csv",
                    index=False,
                )
                test_groups.to_csv(
                    run_directory / "official_test_concept_group_metrics.csv",
                    index=False,
                )
                test_summary["macro_ap"] = float(
                    test_atomic.average_precision.mean()
                )
                test_summary["macro_f1"] = float(test_atomic.f1.mean())
                if bool(config.get("save_probe_latents", True)):
                    test_encoded = encoder.predict(
                        arrays["official_test"], batch_size=batch_size, verbose=0
                    )
                    test_residual = test_encoded.get("residual")
                    np.savez_compressed(
                        run_directory / "official_test_probe_latents.npz",
                        residual=(
                            np.asarray([])
                            if test_residual is None
                            else test_residual
                        ),
                        concept_probabilities=test_probabilities,
                        hard_concepts=(
                            test_probabilities >= 0.5
                        ).astype(np.float32),
                        labels=labels["official_test"],
                        weights=weights["official_test"],
                        image_indices=splits["official_test"],
                        attribute_ids=np.asarray(selected_ids),
                    )
            save_json(test_summary, run_directory / "official_test_result.json")

    pd.DataFrame(results).to_csv(output_root / "all_runs.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/factorized_capacity_pilot.json")
    arguments = parser.parse_args()
    main(arguments.config)
