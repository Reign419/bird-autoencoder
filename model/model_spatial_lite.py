from tensorflow.keras import layers, Model


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


def build_spatial_lite_autoencoder(
    img_shape=(64, 64, 3),
    latent_channels=4
):
    """
    Spatial-latent autoencoder.

    For 64x64 input:
        encoder output latent map = 8 x 8 x latent_channels

    If latent_channels=4:
        effective latent size = 8 * 8 * 4 = 256

    Important:
        No Flatten.
        No Dense latent vector.
        No U-Net skip connections.
    """

    # =====================
    # Encoder
    # =====================
    encoder_input = layers.Input(shape=img_shape)

    x = conv_bn_act(encoder_input, 64)

    x = down_block(x, 64)       # 64 -> 32
    x = down_block(x, 128)      # 32 -> 16
    x = down_block(x, 256)      # 16 -> 8

    # Spatial latent map: 8 x 8 x latent_channels
    latent = layers.Conv2D(
        latent_channels,
        kernel_size=1,
        padding="same",
        name="latent_feature_map"
    )(x)

    encoder = Model(
        encoder_input,
        latent,
        name="spatial_lite_encoder"
    )

    # =====================
    # Decoder
    # =====================
    decoder_input = layers.Input(
        shape=(8, 8, latent_channels),
        name="spatial_latent_input"
    )

    x = conv_bn_act(decoder_input, 256)
    x = residual_block(x, 256)

    x = up_block(x, 128)        # 8 -> 16
    x = up_block(x, 64)         # 16 -> 32
    x = up_block(x, 32)         # 32 -> 64

    decoder_output = layers.Conv2D(
        3,
        kernel_size=3,
        padding="same",
        activation="sigmoid"
    )(x)

    decoder = Model(
        decoder_input,
        decoder_output,
        name="spatial_lite_decoder"
    )

    autoencoder = Model(
        encoder_input,
        decoder(encoder(encoder_input)),
        name="spatial_lite_autoencoder"
    )

    return autoencoder, encoder, decoder