import os
import json
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import argparse
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
    psnr_metric
    )
from train_utils import get_callbacks
from visualize import (
    save_reconstruction_grid,
    save_loss_curve,
    save_metric_curves,
    save_difference_grid
    )  
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
        for idx, exp in enumerate(base_config["experiments"]):
            merged = dict(exp)
            merged.setdefault("name", f"experiment_{idx:02d}")
            merged.setdefault("loss", base_config.get("loss", "l1_ssim_edge"))
            merged.setdefault("learning_rate", base_config.get("learning_rate", 1e-3))
            merged.setdefault("batch_size", base_config.get("batch_size", 32))
            merged.setdefault("epochs", base_config.get("epochs", 60))
            return_experiment = merged
            experiments.append(return_experiment)
        return experiments

    raise ValueError("Please use experiments in config.json.")

def normalize_experiment(exp):
    model_name = exp["model_name"]

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
        exp["latent_grid_size"] = None
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
            latent_dim=latent_dim
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
            latent_grid_size=latent_grid_size
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
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def get_loss(loss_name, perceptual_weight=0.0):
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
        json.dump(obj, f, indent=4)


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

    seed = base_config["random_state"]
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    dataset_path = base_config["dataset_path"]
    output_path = base_config["output_path"]

    os.makedirs(output_path, exist_ok=True)

    img_size = tuple(base_config["img_size"])
    img_shape = (img_size[0], img_size[1], 3)

    learning_rate = base_config["learning_rate"]
    batch_size = base_config["batch_size"]
    epochs = base_config["epochs"]
    loss_name = base_config["loss"]
    loss_fn = get_loss(loss_name)   

    train_images, val_images = load_cub_images(
        dataset_path=dataset_path,
        img_size=img_size,
        test_size=base_config["test_size"],
        random_state=base_config["random_state"]
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
        training_seed = int(exp.get("training_seed", seed))
        random.seed(training_seed)
        np.random.seed(training_seed)
        tf.keras.utils.set_random_seed(training_seed)

        model_name = exp["model_name"]
        loss_name = exp.get(
            "loss", 
            base_config.get("loss", "l1_ssim_edge")
        )
        perceptual_weight = exp.get(
            "perceptual_weight",
            base_config.get("perceptual_weight", 0.0)
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
        loss_label = loss_name
        if loss_name == "l1_ssim_edge_perceptual":
            pw_label = str(perceptual_weight).replace(".", "p")
            loss_label = f"{loss_name}_pw{pw_label}"
        experiment_name = exp.get("name", model_name).replace(" ", "_")
        run_name = (
            f"{experiment_name}_latent_{latent_label.replace('x', '_')}_"
            f"{loss_label}_{timestamp}"
        )
        run_output_path = os.path.join(output_path, run_name)
        os.makedirs(run_output_path, exist_ok=True)

        print(f"\n===== Training {run_name} =====")

        run_config = {
            "model_name": model_name,
            "experiment_name": exp.get("name"),
            "latent_shape": exp["latent_shape"],
            "effective_latent_size": exp["effective_latent_size"],
            "variant": exp.get("variant"),
            "compressed_dim": exp.get("compressed_dim"),
            "spatial_channels": exp.get("spatial_channels"),
            "mixing_initializer": exp.get("mixing_initializer"),
            "mixing_trainable": exp.get("mixing_trainable"),
            "dataset_path": dataset_path,
            "output_path": run_output_path,
            "img_size": list(img_size),
            "img_shape": list(img_shape),
            "train_size": int(train_images.shape[0]),
            "val_size": int(val_images.shape[0]),
            "test_size": base_config["test_size"],
            "random_state": base_config["random_state"],
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
                "perceptual_metric"
            ],
            "perceptual_weight": perceptual_weight,
            "epochs": epochs,
            "batch_size": batch_size,
            "early_stopping": True,
            "reduce_lr_on_plateau": True,
            "checkpoint": True,
        }


        save_json(
            run_config,
            os.path.join(run_output_path, "config.json")
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
                perceptual_metric
            ]
        )

        callbacks = get_callbacks(
            output_path=run_output_path,
            model_name=model_name,
            latent_dim=latent_label
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
            verbose=1
        )

        best_epoch = int(np.argmin(history.history["val_loss"]))

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
            "model": model_name,
            "latent_shape": exp["latent_shape"],
            "loss_name": loss_name,
            "perceptual_weight": perceptual_weight,
            "effective_latent_size": exp["effective_latent_size"],
            "best_epoch": best_epoch + 1,
            # "batch_size": batch_size,
            "best_val_loss": history.history["val_loss"][best_epoch],
            "best_val_mse": history.history["val_mse_metric"][best_epoch],
            "best_val_l1": history.history["val_l1_metric"][best_epoch],
            "best_val_ssim": history.history["val_ssim_metric"][best_epoch],
            "best_val_ssim_loss": history.history["val_ssim_loss_metric"][best_epoch],
            "best_val_edge": history.history["val_edge_metric"][best_epoch],
            "best_val_psnr": history.history["val_psnr_metric"][best_epoch],
            "best_val_perceptual": history.history["val_perceptual_metric"][best_epoch],

            "final_train_loss": train_eval["loss"],
            "final_train_mse": train_eval["mse_metric"],
            "final_train_l1": train_eval["l1_metric"],
            "final_train_ssim": train_eval["ssim_metric"],
            "final_train_edge": train_eval["edge_metric"],
            "final_train_psnr": train_eval["psnr_metric"],
            "final_train_perceptual": train_eval["perceptual_metric"],
            "final_val_loss": val_eval["loss"],
            "final_val_mse": val_eval["mse_metric"],
            "final_val_l1": val_eval["l1_metric"],
            "final_val_ssim": val_eval["ssim_metric"],
            "final_val_edge": val_eval["edge_metric"],
            "final_val_psnr": val_eval["psnr_metric"],
            "final_val_perceptual": val_eval["perceptual_metric"],
        }

        results.append(result)

        save_json(
            result,
            os.path.join(run_output_path, "result.json")
        )

        save_loss_curve(
            history=history,
            output_path=run_output_path,
            filename=f"loss_{run_name}.png"
        )

        save_metric_curves(
            history=history,
            output_path=run_output_path,
            filename=f"metrics_{run_name}.png"
        )

        save_reconstruction_grid(
            model=model,
            images=train_comparison_images,
            output_path=run_output_path,
            filename=f"train_reconstruction_{run_name}.png",
            title=f"Train Reconstruction: {run_name}"
        )

        save_reconstruction_grid(
            model=model,
            images=val_comparison_images,
            output_path=run_output_path,
            filename=f"val_reconstruction_{run_name}.png",
            title=f"Validation Reconstruction: {run_name}"
        )

        save_difference_grid(
            model=model,
            images=train_comparison_images,
            output_path=run_output_path,
            filename=f"train_difference_{run_name}.png",
            title=f"Train Difference: {run_name}"
        )

        save_difference_grid(
            model=model,
            images=val_comparison_images,
            output_path=run_output_path,
            filename=f"val_difference_{run_name}.png",
            title=f"Validation Difference: {run_name}"
        )

    
    df = pd.DataFrame(results)
    df.to_csv(
        os.path.join(output_path, f"summary_results_{model_name}_{timestamp}.csv"),
        index=False
    )

    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    main(args.config)