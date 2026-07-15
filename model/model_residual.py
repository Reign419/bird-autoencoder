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


def conv_bn_act(x, filters, kernel_size=3, strides=1):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    return x


def residual_block(x, filters):
    shortcut = x

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal"
    )(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal"
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    return x


def down_block(x, filters):
    x = conv_bn_act(x, filters, strides=2)
    x = residual_block(x, filters)
    return x


def up_block(x, filters):
    x = layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear"
    )(x)
    x = conv_bn_act(x, filters)
    x = residual_block(x, filters)
    return x


def build_residual_autoencoder(
    img_shape=(64, 64, 3),
    latent_dim=128,
    base_channels=64,
    latent_grid_size=8,
    max_channels=512,
):
    """
    Residual autoencoder with single Dense latent vector.

    Works for both 64x64 and 128x128 inputs.

    The number of down/up-sampling stages is selected from ``img_shape``.
    With ``latent_grid_size=8``, 64x64 uses three stages and 128x128 uses four.
    """

    h, w, _ = img_shape
    start_h, start_w, n_down = _num_downsample(img_shape, latent_grid_size)
    schedule = [
        min(base_channels * (2 ** i), max_channels)
        for i in range(n_down)
    ]

    # =====================
    # Encoder
    # =====================
    encoder_input = layers.Input(shape=img_shape)

    x = conv_bn_act(encoder_input, base_channels)

    for filters in schedule:
        x = down_block(x, filters)

    x = layers.Flatten()(x)

    # single-layer latent vector
    latent = layers.Dense(
        latent_dim,
        name="latent_vector"
    )(x)

    encoder = Model(
        encoder_input,
        latent,
        name="residual_encoder"
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
        x = up_block(x, filters)

    decoder_output = layers.Conv2D(
        3,
        3,
        padding="same",
        activation="sigmoid"
    )(x)

    decoder = Model(
        decoder_input,
        decoder_output,
        name="residual_decoder"
    )

    autoencoder = Model(
        encoder_input,
        decoder(encoder(encoder_input)),
        name="residual_autoencoder"
    )

    return autoencoder, encoder, decoder