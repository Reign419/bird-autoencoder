"""A single-vector interface that preserves a spatially structured code.

Unlike the original residual-lite dense bottleneck, this model performs its
compression locally with a shared 1x1 convolution. Flatten/reshape only change
the tensor interface; they do not learn a global spatial mixing.
"""

from tensorflow.keras import Model, layers

try:
    from .model_common import (
        conv_bn_act,
        make_channel_schedule,
        residual_block,
        residual_down_block,
        residual_up_block,
        resolve_latent_grid,
    )
except ImportError:
    from model_common import (
        conv_bn_act,
        make_channel_schedule,
        residual_block,
        residual_down_block,
        residual_up_block,
        resolve_latent_grid,
    )


def build_structured_vector_lite_autoencoder(
    img_shape=(64, 64, 3),
    latent_dim=1024,
    latent_grid_size=8,
    base_channels=64,
    max_channels=256,
):
    """Build a spatially compressed autoencoder with a 1-D latent interface.

    For a 64x64 input and an 8x8 grid:

    * latent_dim=1024 corresponds to 8x8x16 before flattening.
    * latent_dim=2048 corresponds to 8x8x32 before flattening.

    The decoder begins with a parameter-free reshape. Therefore this model is
    computationally equivalent to the matching spatial-lite model apart from
    exposing its latent as shape ``(batch, latent_dim)``.
    """
    latent_h, latent_w, n_down = resolve_latent_grid(
        img_shape,
        latent_grid_size,
    )
    spatial_area = latent_h * latent_w
    if latent_dim % spatial_area:
        raise ValueError(
            f"latent_dim={latent_dim} must be divisible by latent grid area "
            f"{latent_h}x{latent_w}={spatial_area}."
        )

    latent_channels = latent_dim // spatial_area
    if latent_channels < 1:
        raise ValueError("latent_dim is too small for the requested latent grid.")

    schedule = make_channel_schedule(n_down, base_channels, max_channels)

    # Encoder: identical spatial compression to spatial_lite, followed only by
    # a parameter-free flatten so the public latent interface is one vector.
    encoder_input = layers.Input(shape=img_shape, name="image")
    x = conv_bn_act(encoder_input, base_channels, kernel_initializer=None)
    for filters in schedule:
        x = residual_down_block(x, filters, kernel_initializer=None)

    latent_map = layers.Conv2D(
        latent_channels,
        1,
        padding="same",
        name="structured_latent_map",
    )(x)
    latent_vector = layers.Flatten(name="structured_latent_vector")(latent_map)
    encoder = Model(
        encoder_input,
        latent_vector,
        name=f"structured_vector_encoder_{latent_dim}",
    )

    # Decoder: reshape restores the known grid ordering. There is deliberately
    # no Dense layer here, so no global mixing is introduced.
    decoder_input = layers.Input(shape=(latent_dim,), name="latent_vector_input")
    x = layers.Reshape(
        (latent_h, latent_w, latent_channels),
        name="restore_structured_latent_map",
    )(decoder_input)
    x = conv_bn_act(x, schedule[-1], kernel_initializer=None)
    x = residual_block(x, schedule[-1], kernel_initializer=None)

    decoder_filters = list(reversed(schedule[:-1])) + [max(base_channels // 2, 1)]
    for filters in decoder_filters:
        x = residual_up_block(x, filters, kernel_initializer=None)

    decoder_output = layers.Conv2D(
        img_shape[-1],
        3,
        padding="same",
        activation="sigmoid",
        name="reconstruction",
    )(x)
    decoder = Model(
        decoder_input,
        decoder_output,
        name=f"structured_vector_decoder_{latent_dim}",
    )

    autoencoder = Model(
        encoder_input,
        decoder(encoder(encoder_input)),
        name=f"structured_vector_lite_{latent_dim}",
    )
    autoencoder.structured_vector_metadata = {
        "latent_dim": latent_dim,
        "latent_grid": (latent_h, latent_w),
        "latent_channels": latent_channels,
    }
    return autoencoder, encoder, decoder