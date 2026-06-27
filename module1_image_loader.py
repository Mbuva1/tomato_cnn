"""
=============================================================
MODULE 1: IMAGE LOADER
Project: Tomato Leaf Disease Detection using CNN from Scratch
Tools: Pure Python + NumPy + Pillow (image reading ONLY)
=============================================================
"""

import os
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────
# CLASSES (10 tomato + 1 non_tomato = 11 total)
# ─────────────────────────────────────────────

TOMATO_CLASSES = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
    "non_tomato"        # ← class 10: rejects cars, cups, other leaves etc.
]

NUM_CLASSES = len(TOMATO_CLASSES)   # 11


# ─────────────────────────────────────────────
# LOAD SINGLE IMAGE
# ─────────────────────────────────────────────

def load_image(filepath):
    """
    Load a JPG or PNG image using Pillow.
    Returns NumPy array (H, W, 3) uint8.
    """
    img = Image.open(filepath).convert("RGB")
    return np.array(img, dtype=np.uint8)


# ─────────────────────────────────────────────
# RESIZE IMAGE (pure NumPy bilinear interpolation)
# ─────────────────────────────────────────────

def resize_image(image, target_size=(128, 128)):
    """
    Resize image to target_size using bilinear interpolation.
    Pure NumPy — no OpenCV or Pillow.

    Args:
        image       : NumPy array (H, W, 3) uint8
        target_size : (new_H, new_W)  — default 128x128

    Returns:
        Resized NumPy array (new_H, new_W, 3) uint8
    """
    src_h, src_w = image.shape[:2]
    dst_h, dst_w = target_size

    scale_h = src_h / dst_h
    scale_w = src_w / dst_w

    dst_y = np.arange(dst_h, dtype=np.float32) * scale_h
    dst_x = np.arange(dst_w, dtype=np.float32) * scale_w

    y0 = np.floor(dst_y).astype(np.int32).clip(0, src_h - 1)
    y1 = (y0 + 1).clip(0, src_h - 1)
    x0 = np.floor(dst_x).astype(np.int32).clip(0, src_w - 1)
    x1 = (x0 + 1).clip(0, src_w - 1)

    fy = (dst_y - y0).reshape(-1, 1, 1)
    fx = (dst_x - x0).reshape(1, -1, 1)

    top    = image[y0][:, x0] * (1 - fx) + image[y0][:, x1] * fx
    bottom = image[y1][:, x0] * (1 - fx) + image[y1][:, x1] * fx
    resized = (top * (1 - fy) + bottom * fy).astype(np.uint8)

    return resized


# ─────────────────────────────────────────────
# LOAD FULL DATASET
# ─────────────────────────────────────────────

def load_dataset(dataset_dir, image_size=(128, 128), max_per_class=None):
    """
    Load all images from the dataset directory.
    Expects one subfolder per class named exactly as in TOMATO_CLASSES.

    Args:
        dataset_dir   : Path to dataset root folder
        image_size    : (H, W) to resize all images to
        max_per_class : Limit images per class (useful for quick tests)

    Returns:
        images  : (N, H, W, 3) float32 normalized [0, 1]
        labels  : (N,) int32 class indices
        classes : list of class names actually found
    """
    images  = []
    labels  = []
    classes_found = []

    print(f"\n[Dataset Loader] Scanning : {dataset_dir}")
    print(f"[Dataset Loader] Image size: {image_size[0]}x{image_size[1]}")
    print(f"[Dataset Loader] Classes   : {NUM_CLASSES} (including non_tomato)\n")

    for class_idx, class_name in enumerate(TOMATO_CLASSES):
        class_dir = os.path.join(dataset_dir, class_name)

        if not os.path.isdir(class_dir):
            print(f"  [SKIP] Not found: {class_name}")
            continue

        classes_found.append(class_name)
        files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if max_per_class:
            files = files[:max_per_class]

        loaded = 0
        for fname in files:
            fpath = os.path.join(class_dir, fname)
            try:
                img = load_image(fpath)
                img = resize_image(img, image_size)
                img = img.astype(np.float32) / 255.0
                images.append(img)
                labels.append(class_idx)
                loaded += 1
            except Exception as e:
                print(f"    [ERROR] {fname}: {e}")

        print(f"  [OK] {class_name}: {loaded} images")

    if not images:
        raise ValueError("No images loaded. Check your dataset path.")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    print(f"\n[Dataset Loader] Total images : {len(images)}")
    print(f"[Dataset Loader] Array shape  : {images.shape}")
    print(f"[Dataset Loader] Classes found: {len(classes_found)}")

    return images, labels, classes_found


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 55)
    print("  MODULE 1: IMAGE LOADER — Quick Test")
    print("=" * 55)
    print(f"\n  Total classes : {NUM_CLASSES}")
    for i, c in enumerate(TOMATO_CLASSES):
        print(f"    {i:2d}. {c}")

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"\n[Test] Loading: {path}")
        img = load_image(path)
        print(f"  Original shape : {img.shape}")
        resized = resize_image(img, (128, 128))
        print(f"  Resized shape  : {resized.shape}")
        print("\n  MODULE 1 TEST PASSED ✓")
    else:
        print("\nUsage: python module1_image_loader.py <path_to_image>")