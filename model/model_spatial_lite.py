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

def build_spatial_lite_autoencoder(
    img_shape=(64, 64, 3),
    latent_channels=4,
    latent_grid_size=8,
    base_channels=64,
    max_channels=256,
):
    """Spatial bottleneck without Flatten, Dense bottleneck, or U-Net skips.

    With the default 8x8 latent grid, the same builder supports both 64x64
    (three down/up-sampling stages) and 128x128 (four stages). Build one model
    per input resolution by passing the corresponding ``img_shape``.
    """
    lh, lw, n_down = resolve_latent_grid(img_shape, latent_grid_size)
    schedule = make_channel_schedule(n_down, base_channels, max_channels)

    inp = layers.Input(shape=img_shape)
    x = conv_bn_act(inp, base_channels, kernel_initializer=None)
    for f in schedule:
        x = residual_down_block(x, f, kernel_initializer=None)

    latent = layers.Conv2D(latent_channels, 1, padding="same", name="latent_feature_map")(x)
    encoder = Model(inp, latent, name=f"spatial_encoder_{lh}x{lw}x{latent_channels}")

    din = layers.Input(shape=(lh, lw, latent_channels))
    x = conv_bn_act(din, schedule[-1], kernel_initializer=None)
    x = residual_block(x, schedule[-1], kernel_initializer=None)
    decoder_filters = list(reversed(schedule[:-1])) + [base_channels // 2]
    for f in decoder_filters:
        x = residual_up_block(x, f, kernel_initializer=None)
    out = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x)

    decoder = Model(din, out, name=f"spatial_decoder_{lh}x{lw}x{latent_channels}")
    autoencoder = Model(inp, decoder(encoder(inp)), name=f"spatial_lite_{lh}x{lw}x{latent_channels}")
    return autoencoder, encoder, decoder
