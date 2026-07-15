"""Shared building blocks for the autoencoder model definitions."""

import math
from tensorflow.keras import layers


def resolve_latent_grid(img_shape, latent_grid_size=8):
    """Return latent height, width, and the required power-of-two stages."""
    h, w = img_shape[:2]
    if isinstance(latent_grid_size, int):
        latent_h = latent_w = latent_grid_size
    else:
        latent_h, latent_w = latent_grid_size

    if h % latent_h or w % latent_w:
        raise ValueError("Image size must be divisible by latent grid size.")

    ratio_h = h // latent_h
    ratio_w = w // latent_w
    if ratio_h != ratio_w or ratio_h <= 0 or ratio_h & (ratio_h - 1):
        raise ValueError("Only power-of-two symmetric scaling is supported.")

    return latent_h, latent_w, int(math.log2(ratio_h))


def make_channel_schedule(num_stages, base_channels=64, max_channels=256):
    """Create a capped channel schedule for encoder downsampling stages."""
    if num_stages < 1:
        raise ValueError("At least one downsampling stage is required.")
    return [
        min(base_channels * (2 ** stage), max_channels)
        for stage in range(num_stages)
    ]


def _conv_kwargs(kernel_initializer):
    kwargs = {
        "padding": "same",
        "use_bias": False,
    }
    if kernel_initializer is not None:
        kwargs["kernel_initializer"] = kernel_initializer
    return kwargs


def conv_bn_act(
    x,
    filters,
    kernel_size=3,
    strides=1,
    kernel_initializer="he_normal",
    negative_slope=0.1,
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        **_conv_kwargs(kernel_initializer),
    )(x)
    x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(negative_slope=negative_slope)(x)


def residual_block(
    x,
    filters,
    kernel_initializer="he_normal",
    negative_slope=0.1,
):
    shortcut = x
    conv_kwargs = _conv_kwargs(kernel_initializer)

    x = layers.Conv2D(filters, 3, **conv_kwargs)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=negative_slope)(x)

    x = layers.Conv2D(filters, 3, **conv_kwargs)(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, **conv_kwargs)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    return layers.LeakyReLU(negative_slope=negative_slope)(x)


def residual_down_block(
    x,
    filters,
    kernel_initializer="he_normal",
    negative_slope=0.1,
):
    x = conv_bn_act(
        x,
        filters,
        strides=2,
        kernel_initializer=kernel_initializer,
        negative_slope=negative_slope,
    )
    return residual_block(
        x,
        filters,
        kernel_initializer=kernel_initializer,
        negative_slope=negative_slope,
    )


def residual_up_block(
    x,
    filters,
    kernel_initializer="he_normal",
    negative_slope=0.1,
):
    x = layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    x = conv_bn_act(
        x,
        filters,
        kernel_initializer=kernel_initializer,
        negative_slope=negative_slope,
    )
    return residual_block(
        x,
        filters,
        kernel_initializer=kernel_initializer,
        negative_slope=negative_slope,
    )


def double_conv_up_block(
    x,
    filters,
    kernel_initializer="he_normal",
    negative_slope=0.1,
):
    """Bilinear upsampling followed by two Conv-BN-activation blocks."""
    x = layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    x = conv_bn_act(
        x,
        filters,
        kernel_initializer=kernel_initializer,
        negative_slope=negative_slope,
    )
    return conv_bn_act(
        x,
        filters,
        kernel_initializer=kernel_initializer,
        negative_slope=negative_slope,
    )