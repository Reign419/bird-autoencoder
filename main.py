import os
import json
import tensorflow as tf
import pandas as pd
import numpy as np

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

from model.model_cnn import build_cnn_autoencoder
from model.model_residual import build_residual_autoencoder
from model.model_residual_lite import build_residual_lite_autoencoder
from model.model_resnet50 import build_resnet50_autoencoder

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("Using GPU:", gpus)
else:
    print("No GPU detected. TensorFlow will use CPU.")


def get_model(model_name, img_shape, latent_dim):
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
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def get_loss(loss_name):
    if loss_name == "l1_ssim":
        return l1_ssim_loss
    elif loss_name == "mse_ssim":
        return mse_ssim_loss
    elif loss_name == "l1_ssim_edge":
        return l1_ssim_edge_loss
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
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def main():
    base_config = load_config("config.json")

    dataset_path = base_config["dataset_path"]
    output_path = base_config["output_path"]

    os.makedirs(output_path, exist_ok=True)

    img_size = tuple(base_config["img_size"])
    img_shape = (img_size[0], img_size[1], 3)

    model_name = base_config["model_name"]
    latent_dims = base_config["latent_dims"]

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

    for latent_dim in latent_dims:
        print(f"\n===== Training {model_name}, latent_dim={latent_dim} =====")

        run_name = f"{model_name}_latent{latent_dim}"
        run_output_path = os.path.join(output_path, run_name)
        os.makedirs(run_output_path, exist_ok=True)

        run_config = {
            "model_name": model_name,
            "latent_dim": latent_dim,
            "dataset_path": dataset_path,
            "output_path": run_output_path,
            "img_size": list(img_size),
            "img_shape": list(img_shape),
            "train_size": int(train_images.shape[0]),
            "val_size": int(val_images.shape[0]),
            "test_size": base_config["test_size"],
            "random_state": base_config["random_state"],
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "loss": loss_name,
            "metrics": [
                "mse_metric",
                "l1_metric",
                "ssim_metric",
                "ssim_loss_metric",
                "edge_metric",
                "psnr_metric"
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

        save_json(
            run_config,
            os.path.join(run_output_path, "config.json")
        )

        model, encoder, decoder = get_model(
            model_name=model_name,
            img_shape=img_shape,
            latent_dim=latent_dim
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
                psnr_metric
            ]
        )

        callbacks = get_callbacks(
            output_path=run_output_path,
            model_name=model_name,
            latent_dim=latent_dim
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
            "latent_dim": latent_dim,
            "best_epoch": best_epoch + 1,
            # "batch_size": batch_size,
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
        os.path.join(output_path, f"summary_results_{model_name}.csv"),
        index=False
    )

    print(df)


if __name__ == "__main__":
    main()