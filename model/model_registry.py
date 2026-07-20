"""Central model registry for Stage 1 experiments.

Public model builder functions remain in their original modules.  This registry
only removes training-entrypoint branching and makes supported models explicit.
"""

from .model_bottleneck_ablation import build_bottleneck_ablation_autoencoder
from .model_cnn import build_cnn_autoencoder
from .model_residual import build_residual_autoencoder
from .model_residual_lite import build_residual_lite_autoencoder
from .model_resnet50 import build_resnet50_autoencoder
from .model_spatial_lite import build_spatial_lite_autoencoder
from .model_structured_vector_lite import build_structured_vector_lite_autoencoder


def _dense_builder(builder, *, img_shape, latent_dim, **_):
    return builder(img_shape=img_shape, latent_dim=latent_dim)


def _residual_lite(*, img_shape, latent_dim, latent_grid_size, experiment, **_):
    return build_residual_lite_autoencoder(
        img_shape=img_shape,
        latent_dim=latent_dim,
        latent_grid_size=latent_grid_size,
        base_channels=experiment.get("base_channels", 64),
        max_channels=experiment.get("max_channels", 256),
    )


def _resnet50(*, img_shape, latent_dim, **_):
    return build_resnet50_autoencoder(
        img_shape=img_shape,
        latent_dim=latent_dim,
        weights=None,
        train_backbone=True,
        feature_layer="conv3_block4_out",
    )


def _spatial(*, img_shape, latent_channels, latent_grid_size, experiment, **_):
    return build_spatial_lite_autoencoder(
        img_shape=img_shape,
        latent_channels=latent_channels,
        latent_grid_size=latent_grid_size,
        base_channels=experiment.get("base_channels", 64),
        max_channels=experiment.get("max_channels", 256),
    )


def _structured(*, img_shape, latent_dim, latent_grid_size, experiment, **_):
    return build_structured_vector_lite_autoencoder(
        img_shape=img_shape,
        latent_dim=latent_dim,
        latent_grid_size=latent_grid_size,
        base_channels=experiment.get("base_channels", 64),
        max_channels=experiment.get("max_channels", 256),
    )


def _ablation(*, img_shape, latent_grid_size, experiment, **_):
    return build_bottleneck_ablation_autoencoder(
        img_shape=img_shape,
        variant=experiment["variant"],
        spatial_channels=experiment.get("spatial_channels", 8),
        compressed_dim=experiment.get("compressed_dim"),
        latent_grid_size=latent_grid_size,
        base_channels=experiment.get("base_channels", 64),
        max_channels=experiment.get("max_channels", 256),
        mixing_initializer=experiment.get("mixing_initializer", "orthogonal"),
        mixing_trainable=experiment.get("mixing_trainable", True),
        permutation_seed=experiment.get("permutation_seed", 42),
    )


MODEL_BUILDERS = {
    "cnn": lambda **kwargs: _dense_builder(build_cnn_autoencoder, **kwargs),
    "residual": lambda **kwargs: _dense_builder(build_residual_autoencoder, **kwargs),
    "residual_lite": _residual_lite,
    "resnet50": _resnet50,
    "spatial_lite": _spatial,
    "structured_vector_lite": _structured,
    "bottleneck_ablation": _ablation,
}


def build_registered_model(
    model_name,
    *,
    img_shape,
    latent_dim=None,
    latent_channels=None,
    latent_grid_size=8,
    experiment=None,
):
    try:
        builder = MODEL_BUILDERS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model_name {model_name!r}; expected one of {sorted(MODEL_BUILDERS)}"
        ) from exc
    return builder(
        img_shape=img_shape,
        latent_dim=latent_dim,
        latent_channels=latent_channels,
        latent_grid_size=latent_grid_size,
        experiment=experiment or {},
    )
