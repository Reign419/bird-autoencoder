import tensorflow as tf


# =========================
# VGG perceptual settings
# =========================
# For speed, 112 is usually enough for 64x64 images.
# If you want the standard ImageNet scale, change this to (224, 224).
_VGG_INPUT_SIZE = (112, 112)

# Use relatively low/mid-level features.
# Low layers: edges/textures
# Mid layers: shapes/parts
_VGG_LAYER_NAMES = [
    "block1_conv2",
    "block2_conv2",
    "block3_conv3",
]

_VGG_MODEL = None


def get_vgg_feature_extractor():
    """
    Frozen VGG16 feature extractor for perceptual distance.

    Important:
    - This is VGG perceptual distance, NOT standard LPIPS.
    - Standard LPIPS requires learned linear weights, usually from the PyTorch lpips package.
    """
    global _VGG_MODEL

    if _VGG_MODEL is None:
        try:
            vgg = tf.keras.applications.VGG16(
                include_top=False,
                weights="imagenet",
                input_shape=(_VGG_INPUT_SIZE[0], _VGG_INPUT_SIZE[1], 3),
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to load ImageNet-pretrained VGG16 weights. "
                "If the server has no internet, download/cache the weights first. "
                "Do not use weights=None for real perceptual-loss experiments, "
                "because random VGG features do not provide meaningful perceptual distance."
            ) from e

        vgg.trainable = False

        outputs = [
            vgg.get_layer(name).output
            for name in _VGG_LAYER_NAMES
        ]

        _VGG_MODEL = tf.keras.Model(
            inputs=vgg.input,
            outputs=outputs,
            name="vgg16_perceptual_extractor"
        )

        _VGG_MODEL.trainable = False

    return _VGG_MODEL


def _vgg_preprocess(x):
    """
    Input x should be in [0, 1], RGB.
    VGG16 preprocess_input expects image values in [0, 255].
    """
    x = tf.clip_by_value(x, 0.0, 1.0)
    x = tf.image.resize(x, _VGG_INPUT_SIZE, method="bilinear")
    x = x * 255.0
    x = tf.keras.applications.vgg16.preprocess_input(x)
    return x


def perceptual_metric(y_true, y_pred):
    """
    VGG perceptual feature distance.

    This is NOT true LPIPS.
    It is an unlearned VGG feature distance.
    Lower is better.
    """
    vgg = get_vgg_feature_extractor()

    y_true_vgg = _vgg_preprocess(y_true)
    y_pred_vgg = _vgg_preprocess(y_pred)

    true_features = vgg(y_true_vgg, training=False)
    pred_features = vgg(y_pred_vgg, training=False)

    if not isinstance(true_features, list):
        true_features = [true_features]
        pred_features = [pred_features]

    losses = []

    for f_true, f_pred in zip(true_features, pred_features):
        # L1 feature distance is usually stable.
        losses.append(tf.reduce_mean(tf.abs(f_true - f_pred)))

    return tf.add_n(losses) / len(losses)


# =========================
# Basic components
# These are duplicated here to avoid circular imports with losses.py.
# =========================
def _l1(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def _ssim_loss(y_true, y_pred):
    ssim = tf.reduce_mean(
        tf.image.ssim(y_true, y_pred, max_val=1.0)
    )
    return 1.0 - ssim


def _edge_loss(y_true, y_pred):
    edge_true = tf.image.sobel_edges(y_true)
    edge_pred = tf.image.sobel_edges(y_pred)
    return tf.reduce_mean(tf.abs(edge_true - edge_pred))


def make_l1_ssim_edge_perceptual_loss(
    perceptual_weight=0.05,
    ssim_weight=0.2,
    edge_weight=0.1,
):
    """
    Factory function for:
        L1 + ssim_weight * (1 - SSIM)
           + edge_weight * Edge
           + perceptual_weight * VGG perceptual distance

    Use this when you want to sweep perceptual_weight.
    """
    def loss_fn(y_true, y_pred):
        l1 = _l1(y_true, y_pred)
        ssim = _ssim_loss(y_true, y_pred)
        edge = _edge_loss(y_true, y_pred)
        perc = perceptual_metric(y_true, y_pred)

        return (
            l1 * 0.5
            + ssim_weight * ssim
            + edge_weight * edge
            + perceptual_weight * perc
        )

    # Keras uses this name in logs/checkpoints.
    safe_weight = str(perceptual_weight).replace(".", "p")
    loss_fn.__name__ = f"l1_ssim_edge_vgg_{safe_weight}"

    return loss_fn