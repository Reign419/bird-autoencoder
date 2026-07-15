from tensorflow.keras import layers, Model

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

ddef build_residual_lite_autoencoder(
    img_shape=(64, 64, 3),
    latent_dim=128,
    base_channels=64,
    latent_grid_size=8,
    max_channels=256,
):
    """
    Residual-lite autoencoder with a single Dense latent vector.

    Works for both 64x64 and 128x128 inputs.

    The number of down/up-sampling stages is selected from ``img_shape``.
    With ``latent_grid_size=8``, 64x64 uses three stages and 128x128 uses four.
    """

    start_h, start_w, n_down = resolve_latent_grid(img_shape, latent_grid_size)
    schedule = make_channel_schedule(n_down, base_channels, max_channels)

    # =====================
    # Encoder
    # =====================
    encoder_input = layers.Input(shape=img_shape)

    x = conv_bn_act(encoder_input, base_channels)

    for filters in schedule:
        x = residual_down_block(x, filters)

    x = layers.Flatten()(x)

    # Single-layer latent vector
    latent = layers.Dense(latent_dim, name="latent_vector")(x)

    encoder = Model(
        encoder_input,
        latent,
        name="residual_lite_encoder"
    )

    # =====================
    # Decoder
    # =====================
    decoder_input = layers.Input(shape=(latent_dim,))

    start_channels = schedule[-1]

    x = layers.Dense(
        start_h * start_w * start_channels
    )(decoder_input)

    x = layers.Reshape(
        (start_h, start_w, start_channels)
    )(x)

    x = residual_block(x, start_channels)

    decoder_filters = list(reversed(schedule[:-1])) + [max(base_channels // 2, 1)]
    for filters in decoder_filters:
        x = residual_up_block(x, filters)

    decoder_output = layers.Conv2D(
        3,
        3,
        padding="same",
        activation="sigmoid"
    )(x)

    decoder = Model(
        decoder_input,
        decoder_output,
        name="residual_lite_decoder"
    )

    autoencoder = Model(
        encoder_input,
        decoder(encoder(encoder_input)),
        name="residual_lite_autoencoder"
    )

    return autoencoder, encoder, decoder