"""Shared convolutional stem/trunk for Stage 1 topology comparisons."""

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


def build_spatial_encoder(
    *,
    img_shape,
    latent_channels,
    latent_grid_size,
    base_channels,
    max_channels,
    latent_layer_name,
    encoder_name,
    input_name=None,
):
    """Build the shared downsampling stem and local 1x1 latent projection."""
    latent_h, latent_w, n_down = resolve_latent_grid(img_shape, latent_grid_size)
    schedule = make_channel_schedule(n_down, base_channels, max_channels)
    encoder_input = layers.Input(shape=img_shape, name=input_name)
    x = conv_bn_act(encoder_input, base_channels, kernel_initializer=None)
    for filters in schedule:
        x = residual_down_block(x, filters, kernel_initializer=None)
    latent = layers.Conv2D(
        latent_channels,
        1,
        padding="same",
        name=latent_layer_name,
    )(x)
    encoder = Model(encoder_input, latent, name=encoder_name)
    return encoder, (latent_h, latent_w), schedule


def build_spatial_decoder(
    *,
    img_shape,
    latent_shape,
    schedule,
    base_channels,
    decoder_name,
    input_name=None,
    output_name=None,
):
    """Build the shared residual convolutional decoder trunk."""
    decoder_input = layers.Input(shape=latent_shape, name=input_name)
    decoder_output = decode_spatial_features(
        decoder_input,
        img_shape=img_shape,
        schedule=schedule,
        base_channels=base_channels,
        output_name=output_name,
    )
    return Model(decoder_input, decoder_output, name=decoder_name)


def decode_spatial_features(
    features,
    *,
    img_shape,
    schedule,
    base_channels,
    output_name=None,
):
    """Decode an already restored spatial tensor with the common trunk."""
    x = conv_bn_act(features, schedule[-1], kernel_initializer=None)
    x = residual_block(x, schedule[-1], kernel_initializer=None)
    decoder_filters = list(reversed(schedule[:-1])) + [max(base_channels // 2, 1)]
    for filters in decoder_filters:
        x = residual_up_block(x, filters, kernel_initializer=None)
    return layers.Conv2D(
        img_shape[-1],
        3,
        padding="same",
        activation="sigmoid",
        name=output_name,
    )(x)
