import os
import tensorflow as tf


def get_callbacks(
    output_path,
    monitor="val_ssim_metric",
    mode="max",
    early_stopping_patience=8,
    reduce_lr_patience=3,
    min_lr=1e-5,
):
    checkpoint_dir = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=early_stopping_patience,
        restore_best_weights=True,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        mode=mode,
        factor=0.5,
        patience=reduce_lr_patience,
        min_lr=min_lr,
        verbose=1
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            checkpoint_dir,
            "best.keras"
        ),
        monitor=monitor,
        mode=mode,
        save_best_only=True,
        save_weights_only=False,
    )

    return [early_stop, reduce_lr, checkpoint]
