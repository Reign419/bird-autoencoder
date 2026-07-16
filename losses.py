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
