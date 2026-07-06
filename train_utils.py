import os
import tensorflow as tf


def get_callbacks(output_path, model_name, latent_dim):
    checkpoint_dir = os.path.join(output_path, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-5,
        verbose=1
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            checkpoint_dir,
            f"best_{model_name}_latent{latent_dim}.keras"
        ),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False
    )

    return [early_stop, reduce_lr, checkpoint]