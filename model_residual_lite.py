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
    x = layers.LeakyReLU(alpha=0.1)(x)
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
    x = layers.LeakyReLU(alpha=0.1)(x)

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
    x = layers.LeakyReLU(alpha=0.1)(x)
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


def build_residual_lite_autoencoder(
    img_shape=(64, 64, 3),
    latent_dim=128,
    base_channels=64
):
    # =====================
    # Encoder
    # =====================
    encoder_input = layers.Input(shape=img_shape)

    x = conv_bn_act(encoder_input, base_channels)

    x = down_block(x, base_channels)          # 64 -> 32
    x = down_block(x, base_channels * 2)      # 32 -> 16
    x = down_block(x, base_channels * 4)      # 16 -> 8

    x = layers.Flatten()(x)

    # Single-layer latent vector
    latent = layers.Dense(latent_dim, name="latent_vector")(x)

    encoder = Model(encoder_input, latent, name="residual_lite_encoder")

    # =====================
    # Decoder
    # =====================
    decoder_input = layers.Input(shape=(latent_dim,))

    x = layers.Dense(8 * 8 * base_channels * 4)(decoder_input)
    x = layers.Reshape((8, 8, base_channels * 4))(x)

    x = residual_block(x, base_channels * 4)

    x = up_block(x, base_channels * 2)        # 8 -> 16
    x = up_block(x, base_channels)            # 16 -> 32
    x = up_block(x, base_channels // 2)       # 32 -> 64

    decoder_output = layers.Conv2D(
        3,
        3,
        padding="same",
        activation="sigmoid"
    )(x)

    decoder = Model(decoder_input, decoder_output, name="residual_lite_decoder")

    autoencoder = Model(
        encoder_input,
        decoder(encoder(encoder_input)),
        name="residual_lite_autoencoder"
    )

    return autoencoder, encoder, decoder