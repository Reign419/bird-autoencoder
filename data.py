import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_cub_images(
    dataset_path,
    img_size=(64, 64),
    test_size=0.2,
    random_state=42
):
    image_files = []

    for folder_name in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder_name)

        if os.path.isdir(folder_path):
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(".jpg"):
                    image_files.append(os.path.join(folder_path, img_name))

    print(f"Total images: {len(image_files)}")

    images = []

    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(img_size)
        img = np.array(img).astype("float32") / 255.0
        images.append(img)

    images = np.array(images)

    print("Dataset shape:", images.shape)

    train_images, val_images = train_test_split(
        images,
        test_size=test_size,
        random_state=random_state
    )

    print(f"Train size: {train_images.shape[0]}")
    print(f"Val size: {val_images.shape[0]}")

    return train_images, val_images