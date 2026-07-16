import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split


def list_cub_images(dataset_path):
    """Return CUB image paths in a filesystem-independent order."""
    image_files = []
    for folder_name in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for img_name in sorted(os.listdir(folder_path)):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(folder_path, img_name))
    return image_files


def load_cub_images(
    dataset_path,
    img_size=(128, 128),
    test_size=0.2,
    random_state=42,
    return_manifest=False,
):
    """Load the legacy deterministic 80/20 pilot split.

    ``random_state`` is the split seed. Training randomness is configured
    separately by the experiment runner. Set ``return_manifest=True`` to also
    receive a dataframe recording the exact image-to-split assignment.
    """
    image_files = list_cub_images(dataset_path)

    print(f"Total images: {len(image_files)}")

    images = []
    for img_path in image_files:
        with Image.open(img_path) as image:
            image = image.convert("RGB").resize(img_size)
            images.append(np.asarray(image, dtype="float32") / 255.0)
    images = np.asarray(images)

    print("Dataset shape:", images.shape)

    indices = np.arange(len(image_files))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    train_images = images[train_indices]
    val_images = images[val_indices]

    print(f"Train size: {train_images.shape[0]}")
    print(f"Val size: {val_images.shape[0]}")

    if not return_manifest:
        return train_images, val_images

    split_by_index = np.full(len(image_files), "", dtype=object)
    split_order = np.full(len(image_files), -1, dtype=int)
    split_by_index[train_indices] = "train"
    split_by_index[val_indices] = "val"
    split_order[train_indices] = np.arange(len(train_indices))
    split_order[val_indices] = np.arange(len(val_indices))
    manifest = pd.DataFrame(
        {
            "image_id": [os.path.splitext(os.path.basename(path))[0] for path in image_files],
            "relative_path": [os.path.relpath(path, dataset_path) for path in image_files],
            "class_name": [os.path.basename(os.path.dirname(path)) for path in image_files],
            "split": split_by_index,
            "split_index": split_order,
        }
    )
    return train_images, val_images, manifest
