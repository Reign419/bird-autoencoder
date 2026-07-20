from tensorflow.keras import Model

try:
    from .model_topology_common import build_spatial_decoder, build_spatial_encoder
except ImportError:
    from model_topology_common import build_spatial_decoder, build_spatial_encoder

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
    encoder, (latent_h, latent_w), schedule = build_spatial_encoder(
        img_shape=img_shape,
        latent_channels=latent_channels,
        latent_grid_size=latent_grid_size,
        base_channels=base_channels,
        max_channels=max_channels,
        latent_layer_name="latent_feature_map",
        encoder_name=f"spatial_encoder_{latent_grid_size}x{latent_grid_size}x{latent_channels}",
    )
    decoder = build_spatial_decoder(
        img_shape=img_shape,
        latent_shape=(latent_h, latent_w, latent_channels),
        schedule=schedule,
        base_channels=base_channels,
        decoder_name=f"spatial_decoder_{latent_h}x{latent_w}x{latent_channels}",
    )
    autoencoder = Model(
        encoder.input,
        decoder(encoder(encoder.input)),
        name=f"spatial_lite_{latent_h}x{latent_w}x{latent_channels}",
    )
    return autoencoder, encoder, decoder
