"""Pure configuration utilities for Stage 1 reconstruction experiments.

This module deliberately has no TensorFlow dependency.  Config expansion and
latent accounting can therefore be validated on login/analysis machines before
allocating a training job.
"""

from __future__ import annotations

from copy import deepcopy


SUPPORTED_MODELS = {
    "cnn",
    "residual",
    "residual_lite",
    "resnet50",
    "spatial_lite",
    "structured_vector_lite",
    "bottleneck_ablation",
}
TOPOLOGY_VARIANTS = {"A", "B", "P", "C_prime", "C", "D"}


def build_experiment_list(base_config):
    """Merge global defaults and deterministically expand training seeds."""
    sources = base_config.get("experiments")
    if sources is None:
        sources = _legacy_experiment_sources(base_config)
    experiments = []
    training_seeds = base_config.get("training_seeds")
    for index, source in enumerate(sources):
        experiment = deepcopy(source)
        experiment.setdefault("name", f"experiment_{index:02d}")
        experiment.setdefault("loss", base_config.get("loss", "l1_ssim_edge"))
        experiment.setdefault("learning_rate", base_config.get("learning_rate", 1e-3))
        experiment.setdefault("batch_size", base_config.get("batch_size", 32))
        experiment.setdefault("epochs", base_config.get("epochs", 60))
        seeds = (
            [experiment["training_seed"]]
            if "training_seed" in experiment
            else training_seeds
        )
        if seeds is None:
            experiments.append(experiment)
            continue
        for seed in seeds:
            seeded = deepcopy(experiment)
            seeded["training_seed"] = int(seed)
            experiments.append(seeded)
    return experiments


def _legacy_experiment_sources(base_config):
    """Translate the original main.py latent list format into experiments."""
    model_name = base_config.get("model_name")
    if not model_name:
        raise ValueError("Config must define experiments or model_name")
    if model_name == "spatial_lite":
        values = base_config.get("latent_channels", base_config.get("latent_dim"))
        key = "latent_channels"
    else:
        values = base_config.get("latent_dims", base_config.get("latent_dim"))
        key = "latent_dim"
    if values is None:
        raise ValueError(f"Legacy {model_name} config is missing {key}")
    if not isinstance(values, (list, tuple)):
        values = [values]
    grid = int(base_config.get("latent_grid_size", 8))
    result = []
    for value in values:
        value = int(value)
        name = (
            f"spatial_{grid}x{grid}x{value}"
            if model_name == "spatial_lite"
            else f"{model_name}_latent{value}"
        )
        result.append(
            {
                "name": name,
                "model_name": model_name,
                key: value,
                "latent_grid_size": grid,
            }
        )
    return result


def normalize_experiment(source):
    """Validate one experiment and add the historical latent metadata fields."""
    experiment = deepcopy(source)
    model_name = experiment.get("model_name")
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model_name {model_name!r}; expected one of "
            f"{sorted(SUPPORTED_MODELS)}"
        )

    if model_name == "structured_vector_lite":
        latent_dim = _positive_int(experiment, "latent_dim")
        grid = _positive_int(experiment, "latent_grid_size", default=8)
        spatial_area = grid * grid
        if latent_dim % spatial_area:
            raise ValueError(
                f"structured_vector_lite latent_dim={latent_dim} must be "
                f"divisible by grid area {spatial_area}."
            )
        channels = latent_dim // spatial_area
        experiment.update(
            latent_dim=latent_dim,
            latent_channels=channels,
            latent_grid_size=grid,
            latent_shape=str(latent_dim),
            effective_latent_size=latent_dim,
            latent_label=f"vector{latent_dim}_from_{grid}x{grid}x{channels}",
        )
        return experiment

    if model_name == "bottleneck_ablation":
        variant = experiment.get("variant")
        if variant not in TOPOLOGY_VARIANTS:
            raise ValueError(
                f"Unknown topology variant {variant!r}; expected one of "
                f"{sorted(TOPOLOGY_VARIANTS)}"
            )
        grid = _positive_int(experiment, "latent_grid_size", default=8)
        channels = _positive_int(experiment, "spatial_channels", default=8)
        full_dim = grid * grid * channels
        compressed_dim = experiment.get("compressed_dim")
        if variant in {"C", "D"}:
            compressed_dim = _positive_int(experiment, "compressed_dim")
            if compressed_dim >= full_dim:
                raise ValueError("compressed_dim must be smaller than the full spatial size")
            if variant == "D" and compressed_dim % (grid * grid):
                raise ValueError(
                    f"For D, compressed_dim must be divisible by latent area {grid * grid}"
                )
        elif compressed_dim is not None:
            compressed_dim = int(compressed_dim)
        experiment.update(
            latent_dim=full_dim if variant == "B" else compressed_dim,
            latent_channels=None if variant == "B" else channels,
            latent_grid_size=grid,
            latent_shape=str(full_dim) if variant == "B" else f"{grid}x{grid}x{channels}",
            effective_latent_size=compressed_dim if variant in {"C", "D"} else full_dim,
            latent_label=f"{variant}_K{compressed_dim if variant in {'C', 'D'} else full_dim}",
        )
        return experiment

    if model_name == "spatial_lite":
        grid = _positive_int(experiment, "latent_grid_size", default=8)
        channels = _positive_int(experiment, "latent_channels")
        latent_shape = f"{grid}x{grid}x{channels}"
        experiment.update(
            latent_dim=None,
            latent_channels=channels,
            latent_grid_size=grid,
            latent_shape=latent_shape,
            effective_latent_size=grid * grid * channels,
            latent_label=latent_shape,
        )
        return experiment

    latent_dim = _positive_int(experiment, "latent_dim")
    experiment.update(
        latent_dim=latent_dim,
        latent_channels=None,
        latent_grid_size=_positive_int(experiment, "latent_grid_size", default=8),
        latent_shape=str(latent_dim),
        effective_latent_size=latent_dim,
        latent_label=str(latent_dim),
    )
    return experiment


def prepare_experiments(base_config):
    experiments = [normalize_experiment(item) for item in build_experiment_list(base_config)]
    default_seed = _default_training_seed(base_config)
    identities = [
        (item["name"], int(item.get("training_seed", default_seed)))
        for item in experiments
    ]
    duplicates = sorted({identity for identity in identities if identities.count(identity) > 1})
    if duplicates:
        raise ValueError(f"Duplicate experiment name/seed pairs: {duplicates}")
    return experiments


def build_run_name(experiment, loss_label, training_seed, timestamp):
    """Preserve the historical Stage 1 output-directory reference format."""
    experiment_name = experiment.get("name", experiment["model_name"]).replace(" ", "_")
    latent_label = experiment["latent_label"].replace("x", "_")
    return (
        f"{experiment_name}_latent_{latent_label}_{loss_label}_"
        f"seed{int(training_seed)}_{timestamp}"
    )


def experiment_rows(base_config):
    """Return a compact, stable table representation for config validation."""
    default_seed = _default_training_seed(base_config)
    return [
        {
            "experiment_name": item["name"],
            "model_name": item["model_name"],
            "variant": item.get("variant"),
            "training_seed": int(item.get("training_seed", default_seed)),
            "latent_shape": item["latent_shape"],
            "effective_latent_size": item["effective_latent_size"],
        }
        for item in prepare_experiments(base_config)
    ]


def _positive_int(mapping, key, default=None):
    value = mapping.get(key, default)
    if value is None:
        raise ValueError(f"{key} is required")
    value = int(value)
    if value < 1:
        raise ValueError(f"{key} must be positive")
    return value


def _default_training_seed(base_config):
    split_seed = base_config.get("split_seed", base_config.get("random_state", 42))
    return int(base_config.get("training_seed", split_seed))
