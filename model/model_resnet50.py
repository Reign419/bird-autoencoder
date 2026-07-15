import tensorflow as tf
from tensorflow.keras import layers, Model


try:
    from .model_common import (
        conv_bn_act,
        double_conv_up_block,
        resolve_latent_grid,
    )
except ImportError:
    from model_common import (
        conv_bn_act,
        double_conv_up_block,
        resolve_latent_grid,
    )


def build_resnet50_autoencoder(
    img_shape=(64, 64, 3),
    latent_dim=128,
    weights=None,
    train_backbone=True,
    feature_layer="conv3_block4_out",
    latent_grid_size=8,
):
    """
    ResNet-50 encoder + single Dense latent vector + decoder.

    Works for both 64x64 and 128x128 inputs.

    feature_layer (native ResNet output before grid normalisation):
        "conv3_block4_out" -> about 8x8 for 64x64, about 16x16 for 128x128
        "conv4_block6_out" -> about 4x4 for 64x64, about 8x8 for 128x128
        "conv5_block3_out" -> about 2x2 for 64x64, about 4x4 for 128x128

    Extra stride-2 convolution blocks reduce a feature map that is larger than
    ``latent_grid_size``. A smaller selected backbone feature map is resized.
    The decoder uses three upsampling stages for 64x64 and four for 128x128
    when the target grid is 8x8.
    """

    start_h, start_w, n_up = resolve_latent_grid(img_shape, latent_grid_size)

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

    feature_map = base_resnet.get_layer(feature_layer).output

    # Normalise all supported backbone feature layers and input resolutions to
    # the same pre-flatten spatial grid. For the default conv3 feature, 128x128
    # gets one extra learned downsampling block (16x16 -> 8x8), while 64x64 is
    # already 8x8.
    feature_h = int(feature_map.shape[1])
    feature_w = int(feature_map.shape[2])
    feature_c = int(feature_map.shape[3])

    while feature_h > start_h and feature_w > start_w:
        if feature_h % 2 or feature_w % 2:
            raise ValueError("Backbone feature size cannot be halved to the target grid.")
        feature_map = conv_bn_act(feature_map, feature_c, strides=2)
        feature_h //= 2
        feature_w //= 2

    if feature_h != start_h or feature_w != start_w:
        feature_map = layers.Resizing(
            start_h,
            start_w,
            interpolation="bilinear",
            name="latent_grid_resize",
        )(feature_map)

    x = layers.Flatten()(feature_map)

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

    if feature_layer not in {
        "conv3_block4_out",
        "conv4_block6_out",
        "conv5_block3_out",
    }:
        raise ValueError(f"Unsupported feature_layer: {feature_layer}")

    start_c = int(feature_map.shape[-1])
    up_filters = [max(start_c // (2 ** (i + 1)), 32) for i in range(n_up)]

    x = layers.Dense(start_h * start_w * start_c)(decoder_input)
    x = layers.Reshape((start_h, start_w, start_c))(x)

    x = conv_bn_act(x, start_c)
    x = conv_bn_act(x, start_c)

    for filters in up_filters:
        x = double_conv_up_block(x, filters)

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