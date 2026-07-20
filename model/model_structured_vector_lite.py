"""A single-vector interface that preserves a spatially structured code.

Unlike the original residual-lite dense bottleneck, this model performs its
compression locally with a shared 1x1 convolution. Flatten/reshape only change
the tensor interface; they do not learn a global spatial mixing.
"""

from tensorflow.keras import Model, layers

try:
    from .model_common import resolve_latent_grid
    from .model_topology_common import build_spatial_encoder, decode_spatial_features
except ImportError:
    from model_common import resolve_latent_grid
    from model_topology_common import build_spatial_encoder, decode_spatial_features


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
    latent_h, latent_w, _ = resolve_latent_grid(
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

    spatial_encoder, _, schedule = build_spatial_encoder(
        img_shape=img_shape,
        latent_channels=latent_channels,
        latent_grid_size=latent_grid_size,
        base_channels=base_channels,
        max_channels=max_channels,
        latent_layer_name="structured_latent_map",
        encoder_name=f"structured_spatial_encoder_{latent_dim}",
        input_name="image",
    )
    latent_vector = layers.Flatten(name="structured_latent_vector")(
        spatial_encoder.output
    )
    encoder = Model(
        spatial_encoder.input,
        latent_vector,
        name=f"structured_vector_encoder_{latent_dim}",
    )

    # Decoder: reshape restores the known grid ordering. There is deliberately
    # no Dense layer here, so no global mixing is introduced.
    decoder_input = layers.Input(shape=(latent_dim,), name="latent_vector_input")
    restored_map = layers.Reshape(
        (latent_h, latent_w, latent_channels),
        name="restore_structured_latent_map",
    )(decoder_input)
    decoder_output = decode_spatial_features(
        restored_map,
        img_shape=img_shape,
        schedule=schedule,
        base_channels=base_channels,
        output_name="reconstruction",
    )
    decoder = Model(
        decoder_input,
        decoder_output,
        name=f"structured_vector_decoder_{latent_dim}",
    )

    autoencoder = Model(
        encoder.input,
        decoder(encoder(encoder.input)),
        name=f"structured_vector_lite_{latent_dim}",
    )
    autoencoder.structured_vector_metadata = {
        "latent_dim": latent_dim,
        "latent_grid": (latent_h, latent_w),
        "latent_channels": latent_channels,
    }
    return autoencoder, encoder, decoder
