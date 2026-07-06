from tensorflow.keras import layers, Model


def build_cnn_autoencoder(
    img_shape=(64, 64, 3),
    latent_dim=128
):
    encoder_input = layers.Input(shape=img_shape)

    x = layers.Conv2D(64, 3, strides=2, padding="same")(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(256, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Flatten()(x)

    # single-layer latent vector
    latent = layers.Dense(latent_dim, name="latent_vector")(x)

    encoder = Model(encoder_input, latent, name="cnn_encoder")

    decoder_input = layers.Input(shape=(latent_dim,))

    x = layers.Dense(8 * 8 * 256)(decoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Reshape((8, 8, 256))(x)

    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
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