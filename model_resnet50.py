import tensorflow as tf
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


def up_block(x, filters):
    x = layers.UpSampling2D(
        size=(2, 2),
        interpolation="bilinear"
    )(x)
    x = conv_bn_act(x, filters)
    x = conv_bn_act(x, filters)
    return x


def build_resnet50_autoencoder(
    img_shape=(64, 64, 3),
    latent_dim=128,
    weights=None,
    train_backbone=True,
    feature_layer="conv3_block4_out"
):
    """
    ResNet-50 encoder + single Dense latent vector + decoder.

    feature_layer:
        "conv3_block4_out" -> about 8x8 feature map for 64x64 input
        "conv4_block6_out" -> about 4x4 feature map
        "conv5_block3_out" -> about 2x2 feature map, usually too compressed
    """

    encoder_input = layers.Input(shape=img_shape)

    x_in = encoder_input

    if weights == "imagenet":
        x_in = layers.Lambda(
            lambda x: tf.keras.applications.resnet50.preprocess_input(x * 255.0)
        )(encoder_input)

    base_resnet = tf.keras.applications.ResNet50(
        include_top=False,
        weights=weights,
        input_tensor=x_in
    )

    base_resnet.trainable = train_backbone

    # Use intermediate feature map instead of final 2x2 output
    feature_map = base_resnet.get_layer(feature_layer).output

    # For 64x64 input and conv3_block4_out, this should be 8x8x512
    x = layers.Flatten()(feature_map)

    # Single-layer latent vector
    latent = layers.Dense(
        latent_dim,
        name="latent_vector"
    )(x)

    encoder = Model(
        encoder_input,
        latent,
        name=f"resnet50_encoder_{feature_layer}"
    )

    # =====================
    # Decoder
    # =====================
    decoder_input = layers.Input(shape=(latent_dim,))

    if feature_layer == "conv3_block4_out":
        start_h, start_w, start_c = 8, 8, 512
        up_filters = [256, 128, 64]      # 8 -> 16 -> 32 -> 64

    elif feature_layer == "conv4_block6_out":
        start_h, start_w, start_c = 4, 4, 1024
        up_filters = [512, 256, 128, 64] # 4 -> 8 -> 16 -> 32 -> 64

    elif feature_layer == "conv5_block3_out":
        start_h, start_w, start_c = 2, 2, 2048
        up_filters = [1024, 512, 256, 128, 64] # 2 -> 4 -> 8 -> 16 -> 32 -> 64

    else:
        raise ValueError(f"Unsupported feature_layer: {feature_layer}")

    x = layers.Dense(start_h * start_w * start_c)(decoder_input)
    x = layers.Reshape((start_h, start_w, start_c))(x)

    x = conv_bn_act(x, start_c)
    x = conv_bn_act(x, start_c)

    for filters in up_filters:
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
        name=f"resnet50_decoder_{feature_layer}"
    )

    autoencoder = Model(
        encoder_input,
        decoder(encoder(encoder_input)),
        name=f"resnet50_autoencoder_{feature_layer}"
    )

    return autoencoder, encoder, decoder