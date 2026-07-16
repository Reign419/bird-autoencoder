import os
import json
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import argparse
import platform
import subprocess
import sys
from datetime import datetime

from data import load_cub_images
from losses import (
    l1_ssim_loss,
    mse_ssim_loss,
    l1_ssim_edge_loss,
    mse_ssim_edge_loss,
    l1_metric,
    mse_metric,
    ssim_metric,
    ssim_loss_metric,
    edge_metric,
    psnr_metric,
    make_reconstruction_loss,
    )
from train_utils import get_callbacks
from visualize import (
    save_reconstruction_grid,
    save_loss_curve,
    save_metric_curves,
    save_difference_grid
    )  
from evaluate import evaluate_per_image
from perceptual import (
    perceptual_metric,
    make_l1_ssim_edge_perceptual_loss,
)

from model.model_cnn import build_cnn_autoencoder
from model.model_residual import build_residual_autoencoder
from model.model_residual_lite import build_residual_lite_autoencoder
from model.model_resnet50 import build_resnet50_autoencoder
from model.model_spatial_lite import build_spatial_lite_autoencoder
from model.model_bottleneck_ablation import build_bottleneck_ablation_autoencoder
from model.model_structured_vector_lite import build_structured_vector_lite_autoencoder

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Using GPU:", gpus)
else:
    print("No GPU detected. TensorFlow will use CPU.")

def build_experiment_list(base_config):
    if "experiments" in base_config:
        experiments = []
        training_seeds = base_config.get("training_seeds")
        for idx, exp in enumerate(base_config["experiments"]):
            merged = dict(exp)
            merged.setdefault("name", f"experiment_{idx:02d}")
            merged.setdefault("loss", base_config.get("loss", "l1_ssim_edge"))
            merged.setdefault("learning_rate", base_config.get("learning_rate", 1e-3))
            merged.setdefault("batch_size", base_config.get("batch_size", 32))
            merged.setdefault("epochs", base_config.get("epochs", 60))
            if training_seeds is None or "training_seed" in merged:
                experiments.append(merged)
            else:
                for training_seed in training_seeds:
                    seeded = dict(merged)
                    seeded["training_seed"] = int(training_seed)
                    experiments.append(seeded)
        return experiments

    raise ValueError("Please use experiments in config.json.")

def normalize_experiment(exp):
    model_name = exp["model_name"]

    if model_name == "structured_vector_lite":
        latent_dim = int(exp["latent_dim"])
        latent_grid_size = int(exp.get("latent_grid_size", 8))
        spatial_area = latent_grid_size * latent_grid_size
        if latent_dim % spatial_area:
            raise ValueError(
                f"structured_vector_lite latent_dim={latent_dim} must be "
                f"divisible by grid area {spatial_area}."
            )
        latent_channels = latent_dim // spatial_area

        exp["latent_dim"] = latent_dim
        exp["latent_channels"] = latent_channels
        exp["latent_grid_size"] = latent_grid_size
        exp["latent_shape"] = str(latent_dim)
        exp["effective_latent_size"] = latent_dim
        exp["latent_label"] = (
            f"vector{latent_dim}_from_"
            f"{latent_grid_size}x{latent_grid_size}x{latent_channels}"
        )
        return exp

    if model_name == "bottleneck_ablation":
        variant = exp["variant"]
        grid = int(exp.get("latent_grid_size", 8))
        channels = int(exp.get("spatial_channels", 8))
        full_dim = grid * grid * channels
        compressed_dim = exp.get("compressed_dim")
        if compressed_dim is not None:
            compressed_dim = int(compressed_dim)

        exp["latent_dim"] = compressed_dim
        exp["latent_channels"] = channels
        exp["latent_grid_size"] = grid
        exp["latent_shape"] = f"{grid}x{grid}x{channels}"
        exp["effective_latent_size"] = (
            compressed_dim if variant in {"C", "D"} else full_dim
        )
        exp["latent_label"] = (
            f"{variant}_K{exp['effective_latent_size']}"
        )
        return exp

    if model_name == "spatial_lite":
        latent_grid_size = int(exp.get("latent_grid_size", 8))
        latent_channels = int(exp["latent_channels"])

        latent_shape = f"{latent_grid_size}x{latent_grid_size}x{latent_channels}"
        effective_latent_size = latent_grid_size * latent_grid_size * latent_channels

        exp["latent_dim"] = None
        exp["latent_channels"] = latent_channels
        exp["latent_grid_size"] = latent_grid_size
        exp["latent_shape"] = latent_shape
        exp["effective_latent_size"] = effective_latent_size
        exp["latent_label"] = latent_shape

        return exp

    else:
        latent_dim = int(exp["latent_dim"])
        exp["latent_dim"] = latent_dim
        exp["latent_channels"] = None
        exp["latent_grid_size"] = int(exp.get("latent_grid_size", 8))
        exp["latent_shape"] = str(latent_dim)
        exp["effective_latent_size"] = latent_dim
        exp["latent_label"] = str(latent_dim)

        return exp


def get_model(
    model_name,
    img_shape,
    latent_dim=None,
    latent_channels=None,
    latent_grid_size=8,
    experiment=None,
):
    if model_name == "cnn":
        return build_cnn_autoencoder(
            img_shape=img_shape,
            latent_dim=latent_dim
        )
    elif model_name == "residual":
        return build_residual_autoencoder(
            img_shape=img_shape,
            latent_dim=latent_dim
        )
    elif model_name == "residual_lite":
        return build_residual_lite_autoencoder(
            img_shape=img_shape,
            latent_dim=latent_dim,
            latent_grid_size=experiment.get("latent_grid_size", 8),
            base_channels=experiment.get("base_channels", 64),
            max_channels=experiment.get("max_channels", 256),
        )
    elif model_name == "resnet50":
        return build_resnet50_autoencoder(
            img_shape=img_shape,
            latent_dim=latent_dim,
            weights=None,
            train_backbone=True,
            feature_layer="conv3_block4_out"
        )
    elif model_name == "spatial_lite":
        return build_spatial_lite_autoencoder(
            img_shape=img_shape,
            latent_channels=latent_channels,
            latent_grid_size=latent_grid_size,
            base_channels=experiment.get("base_channels", 64),
            max_channels=experiment.get("max_channels", 256),
        )
    elif model_name == "bottleneck_ablation":
        return build_bottleneck_ablation_autoencoder(
            img_shape=img_shape,
            variant=experiment["variant"],
            spatial_channels=experiment.get("spatial_channels", 8),
            compressed_dim=experiment.get("compressed_dim"),
            latent_grid_size=latent_grid_size,
            base_channels=experiment.get("base_channels", 64),
            max_channels=experiment.get("max_channels", 256),
            mixing_initializer=experiment.get("mixing_initializer", "orthogonal"),
            mixing_trainable=experiment.get("mixing_trainable", True),
            permutation_seed=experiment.get("permutation_seed", 42),
        )
    elif model_name == "structured_vector_lite":
        return build_structured_vector_lite_autoencoder(
            img_shape=img_shape,
            latent_dim=latent_dim,
            latent_grid_size=latent_grid_size,
            base_channels=experiment.get("base_channels", 64),
            max_channels=experiment.get("max_channels", 256),
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def get_loss(loss_name, perceptual_weight=0.0):
    if isinstance(loss_name, dict):
        return make_reconstruction_loss(
            pixel=loss_name.get("pixel", "l1"),
            pixel_weight=loss_name.get("pixel_weight", loss_name.get("l1_weight", 1.0)),
            ssim_weight=loss_name.get("ssim_weight", 0.0),
            edge_weight=loss_name.get("edge_weight", 0.0),
        )
    if loss_name == "l1_ssim":
        return l1_ssim_loss
    elif loss_name == "mse_ssim":
        return mse_ssim_loss
    elif loss_name == "l1_ssim_edge":
        return l1_ssim_edge_loss
    elif loss_name == "l1_ssim_edge_perceptual":
        if perceptual_weight is None:
            perceptual_weight = 0.05
        return make_l1_ssim_edge_perceptual_loss(
            perceptual_weight=perceptual_weight,
            ssim_weight=0.2,
            edge_weight=0.1,
        )
    elif loss_name == "mse_ssim_edge":
        return mse_ssim_edge_loss
    elif loss_name == "mse":
        return tf.keras.losses.MeanSquaredError()
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(
            obj,
            f,
            indent=4,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        )


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_is_dirty():
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def save_provenance(path, model=None):
    save_json(
        {
            "git_commit": git_commit(),
            "git_dirty": git_is_dirty(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "tensorflow_version": tf.__version__,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "parameter_count": model.count_params() if model is not None else None,
        },
        path,
    )


def save_model_summary(model, path):
    with open(path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

def load_config(config_path="config.json"):
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file {config_path} not found!")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {config_path}: {e}")

def main(config_path="config.json"):
    base_config = load_config(config_path)

    split_seed = int(base_config.get("split_seed", base_config.get("random_state", 42)))
    default_training_seed = int(base_config.get("training_seed", split_seed))
    random.seed(default_training_seed)
    np.random.seed(default_training_seed)
    tf.keras.utils.set_random_seed(default_training_seed)

    dataset_path = base_config["dataset_path"]
    output_path = base_config["output_path"]

    os.makedirs(output_path, exist_ok=True)

    img_size = tuple(base_config["img_size"])
    img_shape = (img_size[0], img_size[1], 3)

    train_images, val_images, split_manifest = load_cub_images(
        dataset_path=dataset_path,
        img_size=img_size,
        test_size=base_config["test_size"],
        random_state=split_seed,
        return_manifest=True,
    )

    val_manifest = (
        split_manifest[split_manifest["split"] == "val"]
        .sort_values("split_index")
        .reset_index(drop=True)
    )

    train_comparison_images = train_images[:10]
    val_comparison_images = val_images[:10]

    results = []

    experiments = [
        normalize_experiment(exp)
        for exp in build_experiment_list(base_config)
    ]

    for exp in experiments:
        # Reset before every build so paired variants use the same training seed.
        # Repeat the full config with additional seeds for uncertainty estimates.
        tf.keras.backend.clear_session()
        training_seed = int(exp.get("training_seed", default_training_seed))
        random.seed(training_seed)
        np.random.seed(training_seed)
        tf.keras.utils.set_random_seed(training_seed)

        model_name = exp["model_name"]
        learning_rate = float(exp.get("learning_rate", base_config.get("learning_rate", 1e-3)))
        batch_size = int(exp.get("batch_size", base_config.get("batch_size", 32)))
        epochs = int(exp.get("epochs", base_config.get("epochs", 60)))
        loss_name = exp.get(
            "loss", 
            base_config.get("loss", "l1_ssim_edge")
        )
        perceptual_weight = exp.get(
            "perceptual_weight",
            base_config.get("perceptual_weight", 0.0)
        )
        track_perceptual_metric = bool(
            exp.get(
                "track_perceptual_metric",
                base_config.get("track_perceptual_metric", False),
            )
        )
        loss_fn = get_loss(
            loss_name,
            perceptual_weight=perceptual_weight
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        model, encoder, decoder = get_model(
            model_name=model_name,
            img_shape=img_shape,
            latent_dim=exp["latent_dim"],
            latent_channels=exp["latent_channels"],
            latent_grid_size=exp.get("latent_grid_size", 8),
            experiment=exp,
        )

        latent_label = exp["latent_label"]  # 例如 "8x8x4"
        if isinstance(loss_name, dict):
            loss_label = loss_fn.name
        else:
            loss_label = loss_name
        if loss_name == "l1_ssim_edge_perceptual":
            pw_label = str(perceptual_weight).replace(".", "p")
            loss_label = f"{loss_name}_pw{pw_label}"
        experiment_name = exp.get("name", model_name).replace(" ", "_")
        run_name = (
            f"{experiment_name}_latent_{latent_label.replace('x', '_')}_"
            f"{loss_label}_seed{training_seed}_{timestamp}"
        )
        run_output_path = os.path.join(output_path, run_name)
        os.makedirs(run_output_path, exist_ok=True)
        curves_path = os.path.join(run_output_path, "curves")
        figures_path = os.path.join(run_output_path, "figures")
        os.makedirs(curves_path, exist_ok=True)
        os.makedirs(figures_path, exist_ok=True)

        print(f"\n===== Training {run_name} =====")

        run_config = {
            "model_name": model_name,
            "experiment_name": exp.get("name"),
            "latent_shape": exp["latent_shape"],
            "effective_latent_size": exp["effective_latent_size"],
            "latent_grid_size": exp.get("latent_grid_size"),
            "latent_channels": exp.get("latent_channels"),
            "base_channels": exp.get("base_channels", 64),
            "max_channels": exp.get("max_channels", 256),
            "variant": exp.get("variant"),
            "compressed_dim": exp.get("compressed_dim"),
            "spatial_channels": exp.get("spatial_channels"),
            "mixing_initializer": exp.get("mixing_initializer"),
            "mixing_trainable": exp.get("mixing_trainable"),
            "permutation_seed": exp.get("permutation_seed"),
            "dataset_path": dataset_path,
            "output_path": run_output_path,
            "img_size": list(img_size),
            "img_shape": list(img_shape),
            "train_size": int(train_images.shape[0]),
            "val_size": int(val_images.shape[0]),
            "test_size": base_config["test_size"],
            "split_seed": split_seed,
            "training_seed": training_seed,
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "loss": loss_name,
            "metrics": [
                "mse_metric",
                "l1_metric",
                "ssim_metric",
                "ssim_loss_metric",
                "edge_metric",
                "psnr_metric",
            ] + (["perceptual_metric"] if track_perceptual_metric else []),
            "perceptual_weight": perceptual_weight,
            "track_perceptual_metric": track_perceptual_metric,
            "epochs": epochs,
            "batch_size": batch_size,
            "early_stopping": True,
            "reduce_lr_on_plateau": True,
            "checkpoint": True,
            "monitor": exp.get("monitor", base_config.get("monitor", "val_ssim_metric")),
            "monitor_mode": exp.get("monitor_mode", base_config.get("monitor_mode", "max")),
        }


        save_json(
            run_config,
            os.path.join(run_output_path, "config.json")
        )
        split_manifest.to_csv(
            os.path.join(run_output_path, "split_manifest.csv"),
            index=False,
        )
        save_provenance(
            os.path.join(run_output_path, "provenance.json"),
            model=model,
        )

        save_model_summary(
            model,
            os.path.join(run_output_path, "model_summary.txt")
        )

        save_model_summary(
            encoder,
            os.path.join(run_output_path, "encoder_summary.txt")
        )

        save_model_summary(
            decoder,
            os.path.join(run_output_path, "decoder_summary.txt")
        )

        compile_metrics = [
            mse_metric,
            l1_metric,
            ssim_metric,
            ssim_loss_metric,
            edge_metric,
            psnr_metric,
        ]
        if track_perceptual_metric:
            compile_metrics.append(perceptual_metric)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=compile_metrics,
        )

        callbacks = get_callbacks(
            output_path=run_output_path,
            monitor=run_config["monitor"],
            mode=run_config["monitor_mode"],
            early_stopping_patience=int(
                exp.get(
                    "early_stopping_patience",
                    base_config.get("early_stopping_patience", 8),
                )
            ),
            reduce_lr_patience=int(
                exp.get("reduce_lr_patience", base_config.get("reduce_lr_patience", 3))
            ),
        )

        csv_logger = tf.keras.callbacks.CSVLogger(
            os.path.join(run_output_path, "history.csv")
        )
        callbacks.append(csv_logger)

        history = model.fit(
            train_images,
            train_images,
            validation_data=(val_images, val_images),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        monitor_values = history.history[run_config["monitor"]]
        if run_config["monitor_mode"] == "max":
            best_epoch = int(np.argmax(monitor_values))
        else:
            best_epoch = int(np.argmin(monitor_values))

        train_eval = model.evaluate(
            train_images,
            train_images,
            batch_size=batch_size,
            verbose=0,
            return_dict=True
        )

        val_eval = model.evaluate(
            val_images,
            val_images,
            batch_size=batch_size,
            verbose=0,
            return_dict=True
        )

        result = {
            "run_name": run_name,
            "model": model_name,
            "experiment_name": exp.get("name"),
            "variant": exp.get("variant"),
            "latent_shape": exp["latent_shape"],
            "loss_name": loss_label,
            "loss_config": loss_name if isinstance(loss_name, dict) else None,
            "perceptual_weight": perceptual_weight,
            "effective_latent_size": exp["effective_latent_size"],
            "split_seed": split_seed,
            "training_seed": training_seed,
            "monitor": run_config["monitor"],
            "monitor_mode": run_config["monitor_mode"],
            "best_epoch": best_epoch + 1,
            # "batch_size": batch_size,
            "best_val_loss": history.history["val_loss"][best_epoch],
            "best_val_mse": history.history["val_mse_metric"][best_epoch],
            "best_val_l1": history.history["val_l1_metric"][best_epoch],
            "best_val_ssim": history.history["val_ssim_metric"][best_epoch],
            "best_val_ssim_loss": history.history["val_ssim_loss_metric"][best_epoch],
            "best_val_edge": history.history["val_edge_metric"][best_epoch],
            "best_val_psnr": history.history["val_psnr_metric"][best_epoch],
            "best_val_perceptual": (
                history.history["val_perceptual_metric"][best_epoch]
                if track_perceptual_metric
                else None
            ),

            "final_train_loss": train_eval["loss"],
            "final_train_mse": train_eval["mse_metric"],
            "final_train_l1": train_eval["l1_metric"],
            "final_train_ssim": train_eval["ssim_metric"],
            "final_train_edge": train_eval["edge_metric"],
            "final_train_psnr": train_eval["psnr_metric"],
            "final_train_perceptual": train_eval.get("perceptual_metric"),
            "final_val_loss": val_eval["loss"],
            "final_val_mse": val_eval["mse_metric"],
            "final_val_l1": val_eval["l1_metric"],
            "final_val_ssim": val_eval["ssim_metric"],
            "final_val_edge": val_eval["edge_metric"],
            "final_val_psnr": val_eval["psnr_metric"],
            "final_val_perceptual": val_eval.get("perceptual_metric"),
        }

        results.append(result)

        save_json(
            result,
            os.path.join(run_output_path, "result.json")
        )

        per_image = evaluate_per_image(
            model,
            val_images,
            batch_size=batch_size,
            metadata=val_manifest,
        )
        per_image.to_csv(
            os.path.join(run_output_path, "per_image_metrics.csv"),
            index=False,
        )

        save_loss_curve(
            history=history,
            output_path=curves_path,
            filename="loss.png"
        )

        save_metric_curves(
            history=history,
            output_path=curves_path,
            filename="metrics.png"
        )

        save_reconstruction_grid(
            model=model,
            images=train_comparison_images,
            output_path=figures_path,
            filename="train_reconstruction.png",
            title=f"Train Reconstruction: {run_name}"
        )

        save_reconstruction_grid(
            model=model,
            images=val_comparison_images,
            output_path=figures_path,
            filename="val_reconstruction.png",
            title=f"Validation Reconstruction: {run_name}"
        )

        save_difference_grid(
            model=model,
            images=train_comparison_images,
            output_path=figures_path,
            filename="train_difference.png",
            title=f"Train Difference: {run_name}"
        )

        save_difference_grid(
            model=model,
            images=val_comparison_images,
            output_path=figures_path,
            filename="val_difference.png",
            title=f"Validation Difference: {run_name}"
        )

    
    df = pd.DataFrame(results)
    summary_name = base_config.get("summary_name", "all_runs.csv")
    df.to_csv(os.path.join(output_path, summary_name), index=False)

    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    main(args.config)
