import tensorflow as tf


def l1_metric(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def mse_metric(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def ssim_metric(y_true, y_pred):
    return tf.reduce_mean(
        tf.image.ssim(y_true, y_pred, max_val=1.0)
    )


def ssim_loss_metric(y_true, y_pred):
    return 1.0 - ssim_metric(y_true, y_pred)


def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(
        tf.image.psnr(y_true, y_pred, max_val=1.0)
    )


def edge_metric(y_true, y_pred):
    """
    Sobel edge difference.
    tf.image.sobel_edges output shape:
    (batch, height, width, channels, 2)
    """
    edge_true = tf.image.sobel_edges(y_true)
    edge_pred = tf.image.sobel_edges(y_pred)
    return tf.reduce_mean(tf.abs(edge_true - edge_pred))


def l1_ssim_loss(y_true, y_pred):
    l1 = l1_metric(y_true, y_pred)
    ssim_loss = ssim_loss_metric(y_true, y_pred)
    return l1 + 0.2 * ssim_loss


def mse_ssim_loss(y_true, y_pred):
    mse = mse_metric(y_true, y_pred)
    ssim_loss = ssim_loss_metric(y_true, y_pred)
    return mse + 0.1 * ssim_loss


def l1_ssim_edge_loss(y_true, y_pred):
    l1 = l1_metric(y_true, y_pred)
    ssim_loss = ssim_loss_metric(y_true, y_pred)
    edge = edge_metric(y_true, y_pred)

    return 0.5 * l1 + 0.2 * ssim_loss + 0.1 * edge


def mse_ssim_edge_loss(y_true, y_pred):
    mse = mse_metric(y_true, y_pred)
    ssim_loss = ssim_loss_metric(y_true, y_pred)
    edge = edge_metric(y_true, y_pred)

    return mse + 0.1 * ssim_loss + 0.1 * edge