import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="bird_autoencoder")
def l1_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


@tf.keras.utils.register_keras_serializable(package="bird_autoencoder")
def mse_metric(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


@tf.keras.utils.register_keras_serializable(package="bird_autoencoder")
def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(
        tf.image.ssim(y_true, y_pred, max_val=1.0)
    )


@tf.keras.utils.register_keras_serializable(package="bird_autoencoder")
def ssim_loss_metric(y_true, y_pred):
    return 1.0 - ssim_metric(y_true, y_pred)


@tf.keras.utils.register_keras_serializable(package="bird_autoencoder")
def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(
        tf.image.psnr(y_true, y_pred, max_val=1.0)
    )


@tf.keras.utils.register_keras_serializable(package="bird_autoencoder")
def edge_metric(y_true, y_pred):
    """
    Sobel edge difference.
    tf.image.sobel_edges output shape:
    (batch, height, width, channels, 2)
    """
    edge_true = tf.image.sobel_edges(y_true)
    edge_pred = tf.image.sobel_edges(y_pred)
    return tf.reduce_mean(tf.abs(edge_true - edge_pred))


@tf.keras.utils.register_keras_serializable(package="bird_autoencoder")
def l1_ssim_loss(y_true, y_pred):
    l1 = l1_metric(y_true, y_pred)
    ssim_loss = ssim_loss_metric(y_true, y_pred)
    return l1 + 0.2 * ssim_loss


@tf.keras.utils.register_keras_serializable(package="bird_autoencoder")
def mse_ssim_loss(y_true, y_pred):
    mse = mse_metric(y_true, y_pred)
    ssim_loss = ssim_loss_metric(y_true, y_pred)
    return mse + 0.1 * ssim_loss


@tf.keras.utils.register_keras_serializable(package="bird_autoencoder")
def l1_ssim_edge_loss(y_true, y_pred):
    l1 = l1_metric(y_true, y_pred)
    ssim_loss = ssim_loss_metric(y_true, y_pred)
    edge = edge_metric(y_true, y_pred)

    return 0.5 * l1 + 0.2 * ssim_loss + 0.1 * edge


@tf.keras.utils.register_keras_serializable(package="bird_autoencoder")
def mse_ssim_edge_loss(y_true, y_pred):
    mse = mse_metric(y_true, y_pred)
    ssim_loss = ssim_loss_metric(y_true, y_pred)
    edge = edge_metric(y_true, y_pred)

    return 0.5 * mse + 0.2 * ssim_loss + 0.1 * edge


@tf.keras.utils.register_keras_serializable(package="bird_autoencoder")
class WeightedReconstructionLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        pixel="l1",
        pixel_weight=1.0,
        ssim_weight=0.0,
        edge_weight=0.0,
        name="weighted_reconstruction_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if pixel not in {"l1", "mse"}:
            raise ValueError("pixel must be either 'l1' or 'mse'")
        self.pixel = pixel
        self.pixel_weight = float(pixel_weight)
        self.ssim_weight = float(ssim_weight)
        self.edge_weight = float(edge_weight)

    def call(self, y_true, y_pred):
        pixel_loss = l1_metric(y_true, y_pred) if self.pixel == "l1" else mse_metric(y_true, y_pred)
        return (
            self.pixel_weight * pixel_loss
            + self.ssim_weight * ssim_loss_metric(y_true, y_pred)
            + self.edge_weight * edge_metric(y_true, y_pred)
        )

    def get_config(self):
        return {
            **super().get_config(),
            "pixel": self.pixel,
            "pixel_weight": self.pixel_weight,
            "ssim_weight": self.ssim_weight,
            "edge_weight": self.edge_weight,
        }


def make_reconstruction_loss(
    pixel="l1",
    pixel_weight=1.0,
    ssim_weight=0.0,
    edge_weight=0.0,
):
    """Build an explicit weighted reconstruction loss for controlled ablations."""
    return WeightedReconstructionLoss(
        pixel=pixel,
        pixel_weight=pixel_weight,
        ssim_weight=ssim_weight,
        edge_weight=edge_weight,
        name=(
            f"{pixel}_pw{float(pixel_weight):g}_"
            f"ssim{float(ssim_weight):g}_edge{float(edge_weight):g}"
        ),
    )


@tf.keras.utils.register_keras_serializable(package="bird_autoencoder")
class MaskedWeightedBinaryCrossentropy(tf.keras.losses.Loss):
    """BCE for packed ``[labels, certainty_weights]`` targets.

    Packing the weights into ``y_true`` avoids Keras reducing the attribute
    axis before applying sample weights.  The model prediction has D columns;
    the target has 2D columns.
    """

    def __init__(
        self,
        positive_weights=None,
        name="masked_weighted_binary_crossentropy",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.positive_weights = (
            None if positive_weights is None else [float(value) for value in positive_weights]
        )

    def call(self, y_true, y_pred):
        labels, weights = tf.split(y_true, 2, axis=-1)
        labels = tf.cast(labels, y_pred.dtype)
        weights = tf.cast(weights, y_pred.dtype)
        prediction = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        if self.positive_weights is None:
            positive_weights = tf.ones_like(prediction)
        else:
            positive_weights = tf.cast(self.positive_weights, prediction.dtype)
        bce = -(
            positive_weights * labels * tf.math.log(prediction)
            + (1.0 - labels) * tf.math.log(1.0 - prediction)
        )
        numerator = tf.reduce_sum(bce * weights, axis=-1)
        denominator = tf.maximum(tf.reduce_sum(weights, axis=-1), 1.0)
        return numerator / denominator

    def get_config(self):
        return {**super().get_config(), "positive_weights": self.positive_weights}


@tf.keras.utils.register_keras_serializable(package="bird_autoencoder")
class MaskedBinaryAccuracy(tf.keras.metrics.Metric):
    def __init__(self, threshold=0.5, name="masked_binary_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.weight = self.add_weight(name="weight", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        labels, weights = tf.split(y_true, 2, axis=-1)
        labels = tf.cast(labels, y_pred.dtype)
        weights = tf.cast(weights, y_pred.dtype)
        predictions = tf.cast(y_pred >= self.threshold, y_pred.dtype)
        matches = tf.cast(tf.equal(labels, predictions), y_pred.dtype)
        self.correct.assign_add(tf.reduce_sum(matches * weights))
        self.weight.assign_add(tf.reduce_sum(weights))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.weight)

    def reset_state(self):
        self.correct.assign(0.0)
        self.weight.assign(0.0)

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold}
