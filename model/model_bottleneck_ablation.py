"""Controlled bottleneck ablations for spatial mixing versus compression."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, initializers, layers

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


VARIANTS = {"A", "P", "C_prime", "C", "D"}


@tf.keras.utils.register_keras_serializable(package="bird_autoencoder")
class FixedPermutation(layers.Layer):
    """Apply one deterministic, non-trainable block permutation per vector."""

    def __init__(self, seed=42, block_size=1, **kwargs):
        super().__init__(**kwargs)
        self.seed = int(seed)
        self.block_size = int(block_size)
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        self._permutation = None

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        if feature_dim is None:
            raise ValueError("FixedPermutation requires a known final dimension")
        feature_dim = int(feature_dim)
        if feature_dim % self.block_size:
            raise ValueError("feature dimension must be divisible by block_size")
        group_count = feature_dim // self.block_size
        group_order = np.random.default_rng(self.seed).permutation(group_count)
        offsets = np.arange(self.block_size)
        permutation = (group_order[:, None] * self.block_size + offsets).reshape(-1)
        self._permutation = tf.constant(permutation, dtype=tf.int32)
        super().build(input_shape)

    def call(self, inputs):
        return tf.gather(inputs, self._permutation, axis=-1)

    def get_config(self):
        return {
            **super().get_config(),
            "seed": self.seed,
            "block_size": self.block_size,
        }


def _mixing_initializer(name):
    if name == "identity":
        return initializers.Identity()
    if name == "orthogonal":
        return initializers.Orthogonal()
    if name == "glorot_uniform":
        return initializers.GlorotUniform()
    raise ValueError(
        "mixing_initializer must be one of: identity, orthogonal, glorot_uniform"
    )


def build_bottleneck_ablation_autoencoder(
    img_shape=(64, 64, 3),
    variant="A",
    spatial_channels=8,
    compressed_dim=None,
    latent_grid_size=8,
    base_channels=64,
    max_channels=256,
    mixing_initializer="orthogonal",
    mixing_trainable=True,
    permutation_seed=42,
):
    """Build A/P/C'/C/D with a shared encoder stem and identical decoder.

    The common representation has shape ``H x W x spatial_channels``.  Let
    ``N = H * W * spatial_channels``.

    A
        Identity spatial bottleneck.
    P
        Flatten -> fixed random spatial-position permutation -> reshape. Every
        feature value and within-position channel order is preserved exactly,
        but the H*W positions are reassigned deterministically.
    C_prime
        Flatten -> Dense(N) -> reshape. This is global mixing without
        dimensional compression.
    C
        Flatten -> Dense(K) -> Dense(N) -> reshape. This is global spatial
        mixing plus a rank-K bottleneck.
    D
        1x1 Conv(C') -> 1x1 Conv(spatial_channels), where
        K = H * W * C'. This has the same number K of bottleneck scalars as C
        but never mixes different spatial positions.

    All bottleneck transforms are linear on purpose: nonlinear activations
    would add another experimental variable. There are no skip connections.
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {sorted(VARIANTS)}")
    if spatial_channels < 1:
        raise ValueError("spatial_channels must be positive")

    latent_h, latent_w, n_down = resolve_latent_grid(img_shape, latent_grid_size)
    schedule = make_channel_schedule(n_down, base_channels, max_channels)
    spatial_area = latent_h * latent_w
    full_dim = spatial_area * spatial_channels

    if variant in {"C", "D"}:
        if compressed_dim is None or compressed_dim < 1:
            raise ValueError(f"variant {variant} requires a positive compressed_dim")
        if compressed_dim >= full_dim:
            raise ValueError("compressed_dim must be smaller than the full spatial size")
    if variant == "D" and compressed_dim % spatial_area:
        raise ValueError(
            f"For D, compressed_dim must be divisible by latent area {spatial_area}"
        )

    encoder_input = layers.Input(shape=img_shape, name="image")
    # Match the existing spatial_lite baseline's initialization and blocks.
    x = conv_bn_act(encoder_input, base_channels, kernel_initializer=None)
    for filters in schedule:
        x = residual_down_block(x, filters, kernel_initializer=None)

    common_map = layers.Conv2D(
        spatial_channels,
        1,
        padding="same",
        name="common_spatial_map",
    )(x)

    if variant == "A":
        bottleneck = layers.Activation("linear", name="bottleneck_A_identity")(common_map)
        effective_dim = full_dim
    elif variant == "P":
        x = layers.Flatten(name="flatten_for_fixed_permutation")(common_map)
        x = FixedPermutation(
            seed=permutation_seed,
            block_size=spatial_channels,
            name="bottleneck_P_fixed_permutation",
        )(x)
        bottleneck = layers.Reshape(
            (latent_h, latent_w, spatial_channels),
            name="reshape_after_fixed_permutation",
        )(x)
        effective_dim = full_dim
    elif variant == "C_prime":
        x = layers.Flatten(name="flatten_for_global_mixing")(common_map)
        x = layers.Dense(
            full_dim,
            use_bias=False,
            kernel_initializer=_mixing_initializer(mixing_initializer),
            trainable=mixing_trainable,
            name="bottleneck_C_prime_global_mixing",
        )(x)
        bottleneck = layers.Reshape(
            (latent_h, latent_w, spatial_channels),
            name="reshape_after_global_mixing",
        )(x)
        effective_dim = full_dim
    elif variant == "C":
        x = layers.Flatten(name="flatten_for_global_compression")(common_map)
        latent_vector = layers.Dense(
            compressed_dim,
            use_bias=False,
            name="bottleneck_C_vector",
        )(x)
        x = layers.Dense(
            full_dim,
            use_bias=False,
            name="expand_C_to_spatial_map",
        )(latent_vector)
        bottleneck = layers.Reshape(
            (latent_h, latent_w, spatial_channels),
            name="reshape_after_global_compression",
        )(x)
        effective_dim = compressed_dim
    else:  # D
        compressed_channels = compressed_dim // spatial_area
        spatial_latent = layers.Conv2D(
            compressed_channels,
            1,
            padding="same",
            use_bias=False,
            name="bottleneck_D_spatial_channels",
        )(common_map)
        bottleneck = layers.Conv2D(
            spatial_channels,
            1,
            padding="same",
            use_bias=False,
            name="expand_D_channels",
        )(spatial_latent)
        effective_dim = compressed_dim

    encoder = Model(encoder_input, bottleneck, name=f"ablation_{variant}_encoder")

    decoder_input = layers.Input(
        shape=(latent_h, latent_w, spatial_channels),
        name="decoder_spatial_input",
    )
    x = conv_bn_act(decoder_input, schedule[-1], kernel_initializer=None)
    x = residual_block(x, schedule[-1], kernel_initializer=None)
    decoder_filters = list(reversed(schedule[:-1])) + [max(base_channels // 2, 1)]
    for filters in decoder_filters:
        x = residual_up_block(x, filters, kernel_initializer=None)
    decoder_output = layers.Conv2D(
        img_shape[-1],
        3,
        padding="same",
        activation="sigmoid",
        name="reconstruction",
    )(x)
    decoder = Model(decoder_input, decoder_output, name="shared_ablation_decoder")

    autoencoder = Model(
        encoder_input,
        decoder(encoder(encoder_input)),
        name=f"bottleneck_ablation_{variant}",
    )
    autoencoder.ablation_metadata = {
        "variant": variant,
        "common_shape": (latent_h, latent_w, spatial_channels),
        "full_dim": full_dim,
        "effective_dim": effective_dim,
        "permutation_seed": permutation_seed if variant == "P" else None,
    }
    return autoencoder, encoder, decoder
