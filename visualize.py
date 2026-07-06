import os
import numpy as np
import matplotlib.pyplot as plt


def save_loss_curve(history, output_path, filename):
    """
    Save total train/val loss curve.
    """
    os.makedirs(output_path, exist_ok=True)

    plt.figure(figsize=(6, 4))

    if "loss" in history.history:
        plt.plot(history.history["loss"], label="train_loss")

    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved loss curve to: {save_path}")


def save_metric_curves(history, output_path, filename):
    """
    Save curves for multiple metrics:
    loss, mse, l1, ssim, edge, psnr.
    """
    os.makedirs(output_path, exist_ok=True)

    metric_pairs = [
        ("loss", "val_loss"),
        ("mse_metric", "val_mse_metric"),
        ("l1_metric", "val_l1_metric"),
        ("ssim_metric", "val_ssim_metric"),
        ("edge_metric", "val_edge_metric"),
        ("psnr_metric", "val_psnr_metric"),
    ]

    for train_key, val_key in metric_pairs:
        if train_key not in history.history or val_key not in history.history:
            continue

        plt.figure(figsize=(6, 4))
        plt.plot(history.history[train_key], label=train_key)
        plt.plot(history.history[val_key], label=val_key)
        plt.xlabel("Epoch")
        plt.ylabel(train_key)
        plt.title(f"{train_key} Curve")
        plt.legend()
        plt.tight_layout()

        name = filename.replace(".png", f"_{train_key}.png")
        save_path = os.path.join(output_path, name)
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"Saved metric curve to: {save_path}")


def save_reconstruction_grid(model, images, output_path, filename, n=10, title=None):
    """
    Save original vs reconstructed image grid.
    Top row: original images
    Bottom row: reconstructed images
    """
    os.makedirs(output_path, exist_ok=True)

    images = images[:n]
    reconstructed = model.predict(images, verbose=0)

    plt.figure(figsize=(1.5 * n, 3))

    for i in range(n):
        plt.subplot(2, n, i + 1)
        plt.imshow(np.clip(images[i], 0, 1))
        plt.axis("off")
        if i == 0:
            plt.ylabel("Original")

        plt.subplot(2, n, i + 1 + n)
        plt.imshow(np.clip(reconstructed[i], 0, 1))
        plt.axis("off")
        if i == 0:
            plt.ylabel("Recon")

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()

    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved reconstruction grid to: {save_path}")


def save_difference_grid(model, images, output_path, filename, n=10, title=None):
    """
    Save original, reconstructed, and absolute difference map.
    Row 1: original
    Row 2: reconstructed
    Row 3: absolute difference
    """
    os.makedirs(output_path, exist_ok=True)

    images = images[:n]
    reconstructed = model.predict(images, verbose=0)
    diff = np.abs(images - reconstructed)

    plt.figure(figsize=(1.5 * n, 4.5))

    for i in range(n):
        plt.subplot(3, n, i + 1)
        plt.imshow(np.clip(images[i], 0, 1))
        plt.axis("off")
        if i == 0:
            plt.ylabel("Original")

        plt.subplot(3, n, i + 1 + n)
        plt.imshow(np.clip(reconstructed[i], 0, 1))
        plt.axis("off")
        if i == 0:
            plt.ylabel("Recon")

        plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(np.clip(diff[i] * 4.0, 0, 1))
        plt.axis("off")
        if i == 0:
            plt.ylabel("Diff x4")

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()

    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved difference grid to: {save_path}")