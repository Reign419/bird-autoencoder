import math
from tensorflow.keras import layers, Model


def conv_bn_act(x, filters, kernel_size=3, strides=1):
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(negative_slope=0.1)(x)


def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    return layers.LeakyReLU(negative_slope=0.1)(x)


def down_block(x, filters):
    return residual_block(conv_bn_act(x, filters, strides=2), filters)


def up_block(x, filters):
    x = layers.UpSampling2D((2, 2), interpolation="bilinear")(x)
    return residual_block(conv_bn_act(x, filters), filters)


def _num_downsample(img_shape, latent_grid_size):
    h, w = img_shape[:2]
    if isinstance(latent_grid_size, int):
        lh = lw = latent_grid_size
    else:
        lh, lw = latent_grid_size
    if h % lh or w % lw:
        raise ValueError("Image size must be divisible by latent grid size")
    ratio = h // lh
    if ratio != w // lw or ratio <= 0 or ratio & (ratio - 1):
        raise ValueError("Only power-of-two symmetric downsampling is supported")
    return lh, lw, int(math.log2(ratio))


def build_spatial_lite_autoencoder(
    img_shape=(64, 64, 3),
    latent_channels=4,
    latent_grid_size=8,
    base_channels=64,
    max_channels=256,
):
    """Spatial bottleneck without Flatten, Dense bottleneck, or U-Net skips."""
    lh, lw, n_down = _num_downsample(img_shape, latent_grid_size)
    schedule = [min(base_channels * (2 ** i), max_channels) for i in range(n_down)]

    inp = layers.Input(shape=img_shape)
    x = conv_bn_act(inp, base_channels)
    for f in schedule:
        x = down_block(x, f)

    latent = layers.Conv2D(latent_channels, 1, padding="same", name="latent_feature_map")(x)
    encoder = Model(inp, latent, name=f"spatial_encoder_{lh}x{lw}x{latent_channels}")

    din = layers.Input(shape=(lh, lw, latent_channels))
    x = conv_bn_act(din, schedule[-1])
    x = residual_block(x, schedule[-1])
    decoder_filters = list(reversed(schedule[:-1])) + [base_channels // 2]
    for f in decoder_filters:
        x = up_block(x, f)
    out = layers.Conv2D(3, 3, padding="same", activation="sigmoid")(x)

    decoder = Model(din, out, name=f"spatial_decoder_{lh}x{lw}x{latent_channels}")
    autoencoder = Model(inp, decoder(encoder(inp)), name=f"spatial_lite_{lh}x{lw}x{latent_channels}")
    return autoencoder, encoder, decoder
