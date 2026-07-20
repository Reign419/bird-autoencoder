"""Concept/residual factorized autoencoders for the CUB attribute study."""

from __future__ import annotations

import tensorflow as tf
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


@tf.keras.utils.register_keras_serializable(package="bird_autoencoder")
class SemanticBottleneck(layers.Layer):
    """Hard Bernoulli concepts with STE or a Binary-Concrete fallback."""

    def __init__(self, method="ste", temperature=1.0, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        if method not in {"ste", "gumbel"}:
            raise ValueError("method must be 'ste' or 'gumbel'")
        self.method = method
        self.initial_temperature = float(temperature)
        self.threshold = float(threshold)

    def build(self, input_shape):
        self.temperature = self.add_weight(
            name="temperature",
            shape=(),
            initializer=tf.keras.initializers.Constant(self.initial_temperature),
            trainable=False,
        )

    def call(self, probabilities, training=None):
        probabilities = tf.clip_by_value(probabilities, 1e-6, 1.0 - 1e-6)

        def train_value():
            uniform = tf.random.uniform(tf.shape(probabilities), 1e-6, 1.0 - 1e-6)
            if self.method == "ste":
                hard = tf.cast(probabilities > uniform, probabilities.dtype)
                return probabilities + tf.stop_gradient(hard - probabilities)
            logistic_noise = tf.math.log(uniform) - tf.math.log1p(-uniform)
            logits = tf.math.log(probabilities) - tf.math.log1p(-probabilities)
            return tf.sigmoid((logits + logistic_noise) / self.temperature)

        def test_value():
            return tf.cast(probabilities >= self.threshold, probabilities.dtype)

        if training is None:
            training = False
        if isinstance(training, bool):
            return train_value() if training else test_value()
        return tf.cond(tf.cast(training, tf.bool), train_value, test_value)

    def get_config(self):
        return {
            **super().get_config(),
            "method": self.method,
            "temperature": self.initial_temperature,
            "threshold": self.threshold,
        }


@tf.keras.utils.register_keras_serializable(package="bird_autoencoder")
class ChannelMask(layers.Layer):
    """Apply a fixed non-trainable prefix mask to a fixed-width residual map.

    The residual head and decoder always use ``max_channels`` channels. Capacity
    changes only through this constant 0/1 mask, so trainable parameter counts
    remain identical across the residual-capacity sweep.
    """

    def __init__(self, active_channels, max_channels=15, **kwargs):
        super().__init__(trainable=False, **kwargs)
        self.active_channels = int(active_channels)
        self.max_channels = int(max_channels)
        if self.max_channels <= 0:
            raise ValueError("max_channels must be positive")
        if not 0 <= self.active_channels <= self.max_channels:
            raise ValueError("active_channels must be between 0 and max_channels")

    def build(self, input_shape):
        if int(input_shape[-1]) != self.max_channels:
            raise ValueError(
                f"ChannelMask expected {self.max_channels} channels, got {input_shape[-1]}"
            )
        mask = [1.0] * self.active_channels + [0.0] * (
            self.max_channels - self.active_channels
        )
        self.mask = tf.constant(mask, dtype=self.compute_dtype)[None, None, None, :]

    def call(self, inputs):
        return inputs * tf.cast(self.mask, inputs.dtype)

    def get_config(self):
        return {
            **super().get_config(),
            "active_channels": self.active_channels,
            "max_channels": self.max_channels,
        }


@tf.keras.utils.register_keras_serializable(package="bird_autoencoder")
class ResidualCorruption(layers.Layer):
    """Training-only channel dropout and scale-relative Gaussian noise."""

    def __init__(self, dropout_rate=0.0, noise_std=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dropout_rate = float(dropout_rate)
        self.noise_std = float(noise_std)
        self.spatial_dropout = layers.SpatialDropout2D(self.dropout_rate)

    def call(self, inputs, training=None):
        value = self.spatial_dropout(inputs, training=training)
        if self.noise_std <= 0:
            return value

        def add_noise():
            scale = tf.math.reduce_std(value, axis=(1, 2), keepdims=True)
            scale = tf.maximum(scale, tf.cast(1e-4, value.dtype))
            return value + tf.random.normal(tf.shape(value), dtype=value.dtype) * scale * self.noise_std

        if training is None:
            training = False
        if isinstance(training, bool):
            return add_noise() if training else value
        return tf.cond(tf.cast(training, tf.bool), add_noise, lambda: value)

    def get_config(self):
        return {
            **super().get_config(),
            "dropout_rate": self.dropout_rate,
            "noise_std": self.noise_std,
        }


@tf.keras.utils.register_keras_serializable(package="bird_autoencoder")
class ContinuousControlBottleneck(layers.Layer):
    """Legacy bounded continuous control retained for checkpoint loading only."""

    def __init__(self, noise_std=0.0, **kwargs):
        super().__init__(**kwargs)
        self.noise_std = float(noise_std)
        self.normalization = layers.LayerNormalization(axis=-1)

    def call(self, inputs, training=None):
        bounded = tf.tanh(self.normalization(inputs))
        if self.noise_std <= 0:
            return bounded

        def add_noise():
            return bounded + tf.random.normal(tf.shape(bounded), stddev=self.noise_std)

        if training is None:
            training = False
        if isinstance(training, bool):
            return add_noise() if training else bounded
        return tf.cond(tf.cast(training, tf.bool), add_noise, lambda: bounded)

    def get_config(self):
        return {**super().get_config(), "noise_std": self.noise_std}


class TemperatureAnnealing(tf.keras.callbacks.Callback):
    """Anneal every SemanticBottleneck temperature after each epoch."""

    def __init__(self, start=1.0, end=0.2, epochs=50):
        super().__init__()
        self.start = float(start)
        self.end = float(end)
        self.epochs = max(int(epochs), 1)

    def on_epoch_begin(self, epoch, logs=None):
        fraction = min(max(epoch / max(self.epochs - 1, 1), 0.0), 1.0)
        value = self.start * ((self.end / self.start) ** fraction)
        for layer in self.model.layers:
            if isinstance(layer, SemanticBottleneck):
                layer.temperature.assign(value)


def _build_decoder(
    latent_h,
    latent_w,
    residual_channels,
    condition_dim,
    condition_channels,
    schedule,
    base_channels,
    name,
):
    condition_input = layers.Input(shape=(condition_dim,), name="condition_input")
    condition = layers.Dense(
        latent_h * latent_w * condition_channels,
        name="condition_projection",
    )(condition_input)
    condition = layers.Reshape(
        (latent_h, latent_w, condition_channels),
        name="condition_map",
    )(condition)

    decoder_inputs = [condition_input]
    if residual_channels > 0:
        residual_input = layers.Input(
            shape=(latent_h, latent_w, residual_channels),
            name="residual_input",
        )
        x = layers.Concatenate(axis=-1, name="factorized_latent")([residual_input, condition])
        decoder_inputs = [residual_input, condition_input]
    else:
        x = condition

    x = conv_bn_act(x, schedule[-1], kernel_initializer=None)
    x = residual_block(x, schedule[-1], kernel_initializer=None)
    decoder_filters = list(reversed(schedule[:-1])) + [max(base_channels // 2, 1)]
    for filters in decoder_filters:
        x = residual_up_block(x, filters, kernel_initializer=None)
    output = layers.Conv2D(
        3,
        3,
        padding="same",
        activation="sigmoid",
        name="reconstruction",
    )(x)
    return Model(decoder_inputs, output, name=name)


def build_factorized_lite_autoencoder(
    img_shape=(64, 64, 3),
    concept_dim=224,
    residual_channels=15,
    max_residual_channels=15,
    condition_channels=4,
    latent_grid_size=8,
    base_channels=64,
    max_channels=256,
    mode="concept",
    semantic_method="ste",
    semantic_temperature=1.0,
    residual_dropout=0.1,
    residual_noise_std=0.05,
    control_noise_std=0.0,
):
    """Build concept, matched unsupervised-control, or concept-only models.

    ``residual_channels`` now means the number of active channels in a fixed
    ``max_residual_channels`` residual head. Concept and control conditions both
    follow ``Dense -> sigmoid -> SemanticBottleneck``; the control receives no
    concept supervision. ``control_noise_std`` is accepted only for backward
    config compatibility and is intentionally ignored by the matched control.
    """
    if mode not in {"concept", "control", "concept_only"}:
        raise ValueError("mode must be concept, control, or concept_only")
    if concept_dim <= 0:
        raise ValueError("concept_dim must be positive")
    if max_residual_channels <= 0:
        raise ValueError("max_residual_channels must be positive")
    if not 0 <= residual_channels <= max_residual_channels:
        raise ValueError("residual_channels must be between 0 and max_residual_channels")

    active_residual_channels = int(residual_channels)
    has_residual = mode != "concept_only"
    decoder_residual_channels = max_residual_channels if has_residual else 0

    latent_h, latent_w, n_down = resolve_latent_grid(img_shape, latent_grid_size)
    schedule = make_channel_schedule(n_down, base_channels, max_channels)
    image_input = layers.Input(shape=img_shape, name="image")
    features = conv_bn_act(image_input, base_channels, kernel_initializer=None)
    for filters in schedule:
        features = residual_down_block(features, filters, kernel_initializer=None)

    pooled = layers.GlobalAveragePooling2D(name="global_pool")(features)
    decoder = _build_decoder(
        latent_h,
        latent_w,
        decoder_residual_channels,
        concept_dim,
        condition_channels,
        schedule,
        base_channels,
        name=f"{mode}_decoder",
    )

    if has_residual:
        residual_full = layers.Conv2D(
            max_residual_channels,
            1,
            padding="same",
            name="residual_latent",
        )(features)
        residual_clean = ChannelMask(
            active_channels=active_residual_channels,
            max_channels=max_residual_channels,
            name="residual_channel_mask",
        )(residual_full)
        residual_train = ResidualCorruption(
            residual_dropout,
            residual_noise_std,
            name="residual_corruption",
        )(residual_clean)

    if mode in {"concept", "concept_only"}:
        logits = layers.Dense(concept_dim, name="concept_logits")(pooled)
        probabilities = layers.Activation("sigmoid", name="concepts")(logits)
        condition = SemanticBottleneck(
            method=semantic_method,
            temperature=semantic_temperature,
            name="semantic_bottleneck",
        )(probabilities)
        reconstruction = decoder([residual_train, condition] if has_residual else condition)
        reconstruction = layers.Activation("linear", name="reconstruction")(reconstruction)
        outputs = {"reconstruction": reconstruction, "concepts": probabilities}
        encoder_outputs = {"concepts": probabilities, "semantic": condition}
        if has_residual:
            encoder_outputs["residual"] = residual_clean
        encoder = Model(image_input, encoder_outputs, name=f"{mode}_encoder")
    else:
        control_logits = layers.Dense(concept_dim, name="control_logits")(pooled)
        control_probabilities = layers.Activation("sigmoid", name="control_probabilities")(
            control_logits
        )
        control = SemanticBottleneck(
            method=semantic_method,
            temperature=semantic_temperature,
            name="control_bottleneck",
        )(control_probabilities)
        reconstruction = decoder([residual_train, control])
        reconstruction = layers.Activation("linear", name="reconstruction")(reconstruction)
        outputs = reconstruction
        encoder = Model(
            image_input,
            {
                "control": control,
                "control_probabilities": control_probabilities,
                "residual": residual_clean,
            },
            name="control_encoder",
        )

    model = Model(image_input, outputs, name=f"factorized_{mode}_autoencoder")
    return model, encoder, decoder
