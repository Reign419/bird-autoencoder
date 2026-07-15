import math
from tensorflow.keras import layers, Model


def _num_downsample(img_shape, latent_grid_size):
    h, w = img_shape[:2]
    if isinstance(latent_grid_size, int):
        lh = lw = latent_grid_size
    else:
        lh, lw = latent_grid_size

    if h % lh or w % lw:
        raise ValueError("Image size must be divisible by latent grid size.")

    ratio_h = h // lh
    ratio_w = w // lw
    if ratio_h != ratio_w or ratio_h <= 0 or ratio_h & (ratio_h - 1):
        raise ValueError("Only power-of-two symmetric downsampling is supported.")

    return lh, lw, int(math.log2(ratio_h))


def build_cnn_autoencoder(
    img_shape=(64, 64, 3),
    latent_dim=128,
    latent_grid_size=8,
    base_channels=64,
    max_channels=256,
):
    """Dense-vector CNN autoencoder with a fixed pre-flatten spatial grid.

    With the default ``latent_grid_size=8``, a 64x64 model uses three
    down/up-sampling stages and a 128x128 model uses four. Build one model per
    input resolution by passing the corresponding ``img_shape``.
    """
    h, w, _ = img_shape
    start_h, start_w, n_down = _num_downsample(img_shape, latent_grid_size)
    schedule = [
        min(base_channels * (2 ** i), max_channels)
        for i in range(n_down)
    ]

    encoder_input = layers.Input(shape=img_shape)

    x = encoder_input
    for filters in schedule:
        x = layers.Conv2D(filters, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.Flatten()(x)

    # single-layer latent vector
    latent = layers.Dense(latent_dim, name="latent_vector")(x)

    encoder = Model(encoder_input, latent, name="cnn_encoder")

    # =====================
    # Decoder
    # =====================
    decoder_input = layers.Input(shape=(latent_dim,))

    start_c = schedule[-1]

    x = layers.Dense(start_h * start_w * start_c)(decoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Reshape((start_h, start_w, start_c))(x)

    for filters in reversed(schedule):
        x = layers.Conv2DTranspose(
            filters, 3, strides=2, padding="same"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    decoder_output = layers.Conv2D(
        3,
        3,
        padding="same",
        activation="sigmoid"
    )(x)

    decoder = Model(decoder_input, decoder_output, name="cnn_decoder")

    autoencoder = Model(
        encoder_input,
        decoder(encoder(encoder_input)),
        name="cnn_autoencoder"
    )

    return autoencoder, encoder, decoder