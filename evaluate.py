"""Evaluation helpers shared by training and standalone analysis."""

import numpy as np
import pandas as pd
import tensorflow as tf


def per_image_reconstruction_metrics(y_true, y_pred):
    """Return one row of reconstruction metrics per image."""
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    image_axes = (1, 2, 3)

    mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=image_axes)
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred), axis=image_axes)
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    psnr = tf.image.psnr(y_true, y_pred, max_val=1.0)

    edge_true = tf.image.sobel_edges(y_true)
    edge_pred = tf.image.sobel_edges(y_pred)
    edge = tf.reduce_mean(
        tf.abs(edge_true - edge_pred),
        axis=(1, 2, 3, 4),
    )

    return pd.DataFrame(
        {
            "mse": mse.numpy(),
            "l1": l1.numpy(),
            "ssim": ssim.numpy(),
            "psnr": psnr.numpy(),
            "edge": edge.numpy(),
        }
    )


def evaluate_per_image(model, images, batch_size=32, metadata=None):
    """Predict images and return per-image metrics with optional metadata."""
    predictions = model.predict(images, batch_size=batch_size, verbose=0)
    metrics = per_image_reconstruction_metrics(images, predictions)
    if metadata is None:
        metrics.insert(0, "sample_index", np.arange(len(metrics)))
        return metrics

    metadata = metadata.reset_index(drop=True)
    if len(metadata) != len(metrics):
        raise ValueError("metadata and images must contain the same number of rows")
    return pd.concat([metadata, metrics], axis=1)

