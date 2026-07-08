import os
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

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
)
from train_utils import get_callbacks
from visualize import (
    save_reconstruction_grid,
    save_loss_curve,
    save_metric_curves,
    save_difference_grid,
)

from model.model_cnn import build_cnn_autoencoder
from model.model_residual import build_residual_autoencoder
from model.model_residual_lite import build_residual_lite_autoencoder
from model.model_resnet50 import build_resnet50_autoencoder
from model.model_spatial_lite import build_spatial_lite_autoencoder


SUPPORTED_MODELS = {"cnn", "residual", "residual_lite", "resnet50", "spatial_lite"}


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def configure_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU:", gpus)
    else:
        print("No GPU detected. TensorFlow will use CPU.")


def get_model(model_name, img_shape, latent_dim=None, latent_channels=None, latent_grid_size=8):
    if model_name == "cnn":
        return build_cnn_autoencoder(
            img_shape=img_shape,
            latent_dim=latent_dim,
        )
    if model_name == "residual":
        return build_residual_autoencoder(
            img_shape=img_shape,
            latent_dim=latent_dim,
        )
    if model_name == "residual_lite":
        return build_residual_lite_autoencoder(
            img_shape=img_shape,
            latent_dim=latent_dim,
        )
    if model_name == "resnet50":
        return build_resnet50_autoencoder(
            img_shape=img_shape,
            latent_dim=latent_dim,
            weights=None,
            train_backbone=True,
            feature_layer="conv3_block4_out",
        )
    if model_name == "spatial_lite":
        return build_spatial_lite_autoencoder(
            img_shape=img_shape,
            latent_channels=latent_channels,
            latent_grid_size=latent_grid_size,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def get_loss(loss_name):
    if loss_name == "mse":
        return tf.keras.losses.MeanSquaredError()
    if loss_name == "l1":
        return tf.keras.losses.MeanAbsoluteError()
    if loss_name == "l1_ssim":
        return l1_ssim_loss
    if loss_name == "mse_ssim":
        return mse_ssim_loss
    if loss_name == "l1_ssim_edge":
        return l1_ssim_edge_loss
    if loss_name == "mse_ssim_edge":
        return mse_ssim_edge_loss

    raise ValueError(f"Unknown loss_name: {loss_name}")


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=4)


def save_model_summary(model, path):
    with open(path, "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))


def load_config(config_path="config.json"):
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Config file {config_path} not found. Copy config.example.json to config.json first."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc


def build_experiment_list(base_config):
    """
    Supports two modes:

    1. New recommended mode:
       "experiments": [
           {"name": "dense256", "model_name": "residual_lite", "latent_dim": 256},
           {"name": "spatial_8x8x4", "model_name": "spatial_lite", "latent_grid_size": 8, "latent_channels": 4}
       ]

    2. Backward-compatible mode:
       "model_name": "residual_lite",
       "latent_dims": [128, 256, 512]
    """
    if "experiments" in base_config:
        experiments = []
        for idx, exp in enumerate(base_config["experiments"]):
            merged = dict(exp)
            merged.setdefault("name", f"experiment_{idx:02d}")
            merged.setdefault("model_name", base_config.get("model_name"))
            merged.setdefault("loss", base_config.get("loss", "l1_ssim_edge"))
            merged.setdefault("learning_rate", base_config.get("learning_rate", 1e-3))
            merged.setdefault("batch_size", base_config.get("batch_size", 32))
            merged.setdefault("epochs", base_config.get("epochs", 70))
            experiments.append(merged)
        return experiments

    model_name = base_config["model_name"]
    experiments = []
    for latent_value in base_config["latent_dims"]:
        if model_name == "spatial_lite":
            experiments.append({
                "name": f"spatial_{base_config.get('latent_grid_size', 8)}x{base_config.get('latent_grid_size', 8)}x{latent_value}",
                "model_name": model_name,
                "latent_grid_size": base_config.get("latent_grid_size", 8),
                "latent_channels": latent_value,
                "loss": base_config.get("loss", "l1_ssim_edge"),
                "learning_rate": base_config.get("learning_rate", 1e-3),
                "batch_size": base_config.get("batch_size", 32),
                "epochs": base_config.get("epochs", 70),
            })
        else:
            experiments.append({
                "name": f"{model_name}_latent{latent_value}",
                "model_name": model_name,
                "latent_dim": latent_value,
                "loss": base_config.get("loss", "l1_ssim_edge"),
                "learning_rate": base_config.get("learning_rate", 1e-3),
                "batch_size": base_config.get("batch_size", 32),
                "epochs": base_config.get("epochs", 70),
            })
    return experiments


def normalize_experiment(exp, img_size):
    model_name = exp["model_name"]
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_name={model_name}. Supported: {sorted(SUPPORTED_MODELS)}")

    normalized = dict(exp)

    if model_name == "spatial_lite":
        latent_grid_size = int(normalized.get("latent_grid_size", 8))
        latent_channels = normalized.get("latent_channels", normalized.get("latent_dim"))
        if latent_channels is None:
            raise ValueError("spatial_lite experiments need latent_channels, e.g. 4 for 8x8x4.")
        latent_channels = int(latent_channels)
        latent_shape = f"{latent_grid_size}x{latent_grid_size}x{latent_channels}"
        effective_latent_size = latent_grid_size * latent_grid_size * latent_channels
        latent_label = latent_shape

        normalized.update({
            "latent_dim": None,
            "latent_channels": latent_channels,
            "latent_grid_size": latent_grid_size,
            "latent_shape": latent_shape,
            "effective_latent_size": effective_latent_size,
            "latent_label": latent_label,
        })
        return normalized

    latent_dim = normalized.get("latent_dim", normalized.get("latent_dim_or_channels"))
    if latent_dim is None:
        raise ValueError(f"{model_name} experiments need latent_dim.")
    latent_dim = int(latent_dim)

    normalized.update({
        "latent_dim": latent_dim,
        "latent_channels": None,
        "latent_grid_size": None,
        "latent_shape": str(latent_dim),
        "effective_latent_size": latent_dim,
        "latent_label": str(latent_dim),
    })
    return normalized


def main():
    base_config = load_config("config.json")

    random_state = int(base_config.get("random_state", 42))
    set_global_seed(random_state)
    configure_gpu_memory_growth()

    dataset_path = base_config["dataset_path"]
    output_path = base_config.get("output_path", "./outputs")
    os.makedirs(output_path, exist_ok=True)

    img_size = tuple(base_config.get("img_size", [64, 64]))
    img_shape = (img_size[0], img_size[1], 3)

    test_size = base_config.get("test_size", 0.2)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_images, val_images = load_cub_images(
        dataset_path=dataset_path,
        img_size=img_size,
        test_size=test_size,
        random_state=random_state,
    )

    train_comparison_images = train_images[:10]
    val_comparison_images = val_images[:10]

    experiments = [
        normalize_experiment(exp, img_size=img_size)
        for exp in build_experiment_list(base_config)
    ]

    results = []

    for exp in experiments:
        model_name = exp["model_name"]
        loss_name = exp.get("loss", base_config.get("loss", "l1_ssim_edge"))
        loss_fn = get_loss(loss_name)
        learning_rate = float(exp.get("learning_rate", base_config.get("learning_rate", 1e-3)))
        batch_size = int(exp.get("batch_size", base_config.get("batch_size", 32)))
        epochs = int(exp.get("epochs", base_config.get("epochs", 70)))

        run_name = f"{exp['name']}_{loss_name}_{timestamp}"
        run_output_path = os.path.join(output_path, run_name)
        os.makedirs(run_output_path, exist_ok=True)

        print(
            f"\n===== Training {model_name}, latent={exp['latent_shape']}, "
            f"effective_size={exp['effective_latent_size']}, loss={loss_name} ====="
        )

        run_config = {
            "run_name": run_name,
            "model_name": model_name,
            "latent_dim": exp["latent_dim"],
            "latent_channels": exp["latent_channels"],
            "latent_grid_size": exp["latent_grid_size"],
            "latent_shape": exp["latent_shape"],
            "effective_latent_size": exp["effective_latent_size"],
            "dataset_path": dataset_path,
            "output_path": run_output_path,
            "img_size": list(img_size),
            "img_shape": list(img_shape),
            "train_size": int(train_images.shape[0]),
            "val_size": int(val_images.shape[0]),
            "test_size": test_size,
            "random_state": random_state,
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
            ],
            "epochs": epochs,
            "batch_size": batch_size,
            "early_stopping": True,
            "reduce_lr_on_plateau": True,
            "checkpoint": True,
        }

        if model_name == "resnet50":
            run_config["resnet50_weights"] = None
            run_config["train_backbone"] = True
            run_config["feature_layer"] = "conv3_block4_out"

        save_json(run_config, os.path.join(run_output_path, "config.json"))

        model, encoder, decoder = get_model(
            model_name=model_name,
            img_shape=img_shape,
            latent_dim=exp["latent_dim"],
            latent_channels=exp["latent_channels"],
            latent_grid_size=exp["latent_grid_size"] or 8,
        )

        save_model_summary(model, os.path.join(run_output_path, "model_summary.txt"))
        save_model_summary(encoder, os.path.join(run_output_path, "encoder_summary.txt"))
        save_model_summary(decoder, os.path.join(run_output_path, "decoder_summary.txt"))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=[
                mse_metric,
                l1_metric,
                ssim_metric,
                ssim_loss_metric,
                edge_metric,
                psnr_metric,
            ],
        )

        callbacks = get_callbacks(
            output_path=run_output_path,
            model_name=model_name,
            latent_dim=exp["latent_label"],
        )

        csv_logger = tf.keras.callbacks.CSVLogger(
            os.path.join(run_output_path, f"log_{run_name}.csv")
        )
        callbacks.append(csv_logger)

        history = model.fit(
            train_images,
            train_images,
            validation_data=(val_images, val_images),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        best_epoch = int(np.argmin(history.history["val_loss"]))

        train_eval = model.evaluate(
            train_images,
            train_images,
            batch_size=batch_size,
            verbose=0,
            return_dict=True,
        )
        val_eval = model.evaluate(
            val_images,
            val_images,
            batch_size=batch_size,
            verbose=0,
            return_dict=True,
        )

        result = {
            "run_name": run_name,
            "model": model_name,
            "loss": loss_name,
            "latent_shape": exp["latent_shape"],
            "latent_dim_or_channels": exp["latent_channels"] if model_name == "spatial_lite" else exp["latent_dim"],
            "effective_latent_size": exp["effective_latent_size"],
            "best_epoch": best_epoch + 1,
            "best_val_loss": history.history["val_loss"][best_epoch],
            "best_val_mse": history.history["val_mse_metric"][best_epoch],
            "best_val_l1": history.history["val_l1_metric"][best_epoch],
            "best_val_ssim": history.history["val_ssim_metric"][best_epoch],
            "best_val_ssim_loss": history.history["val_ssim_loss_metric"][best_epoch],
            "best_val_edge": history.history["val_edge_metric"][best_epoch],
            "best_val_psnr": history.history["val_psnr_metric"][best_epoch],
            "final_train_loss": train_eval["loss"],
            "final_train_mse": train_eval["mse_metric"],
            "final_train_l1": train_eval["l1_metric"],
            "final_train_ssim": train_eval["ssim_metric"],
            "final_train_edge": train_eval["edge_metric"],
            "final_train_psnr": train_eval["psnr_metric"],
            "final_val_loss": val_eval["loss"],
            "final_val_mse": val_eval["mse_metric"],
            "final_val_l1": val_eval["l1_metric"],
            "final_val_ssim": val_eval["ssim_metric"],
            "final_val_edge": val_eval["edge_metric"],
            "final_val_psnr": val_eval["psnr_metric"],
        }

        results.append(result)
        save_json(result, os.path.join(run_output_path, "result.json"))

        save_loss_curve(
            history=history,
            output_path=run_output_path,
            filename=f"loss_{run_name}.png",
        )
        save_metric_curves(
            history=history,
            output_path=run_output_path,
            filename=f"metrics_{run_name}.png",
        )
        save_reconstruction_grid(
            model=model,
            images=train_comparison_images,
            output_path=run_output_path,
            filename=f"train_reconstruction_{run_name}.png",
            title=f"Train Reconstruction: {run_name}",
        )
        save_reconstruction_grid(
            model=model,
            images=val_comparison_images,
            output_path=run_output_path,
            filename=f"val_reconstruction_{run_name}.png",
            title=f"Validation Reconstruction: {run_name}",
        )
        save_difference_grid(
            model=model,
            images=train_comparison_images,
            output_path=run_output_path,
            filename=f"train_difference_{run_name}.png",
            title=f"Train Difference: {run_name}",
        )
        save_difference_grid(
            model=model,
            images=val_comparison_images,
            output_path=run_output_path,
            filename=f"val_difference_{run_name}.png",
            title=f"Validation Difference: {run_name}",
        )

    df = pd.DataFrame(results)
    summary_path = os.path.join(output_path, f"summary_results_{timestamp}.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nSaved summary to: {summary_path}")
    print(df)


if __name__ == "__main__":
    main()
