"""
=============================================================
MODULE 2: PREPROCESSING
Project: Tomato Leaf Disease Detection using CNN from Scratch
Tools: Pure Python + NumPy only
=============================================================

Handles:
  - Shuffling the dataset
  - Splitting into Train / Validation / Test sets
  - One-hot encoding labels
  - Normalizing pixel values
  - Data augmentation (flip, brightness, rotation)
=============================================================
"""

import numpy as np
from module1_image_loader import load_dataset, TOMATO_CLASSES, NUM_CLASSES


# ─────────────────────────────────────────────
# 1. SHUFFLE
# ─────────────────────────────────────────────

def shuffle_dataset(images, labels, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(len(images))
    return images[idx], labels[idx]


# ─────────────────────────────────────────────
# 2. SPLIT
# ─────────────────────────────────────────────

def split_dataset(images, labels, train=0.70, val=0.15, test=0.15):
    """Split into train / val / test sets."""
    assert abs(train + val + test - 1.0) < 1e-6, "Fractions must sum to 1.0"

    N         = len(images)
    train_end = int(N * train)
    val_end   = int(N * (train + val))

    print(f"\n[Split] Total      : {N}")
    print(f"[Split] Train      : {train_end}  ({train*100:.0f}%)")
    print(f"[Split] Validation : {val_end - train_end}  ({val*100:.0f}%)")
    print(f"[Split] Test       : {N - val_end}  ({test*100:.0f}%)")

    return (
        images[:train_end],  labels[:train_end],
        images[train_end:val_end], labels[train_end:val_end],
        images[val_end:],    labels[val_end:]
    )


# ─────────────────────────────────────────────
# 3. ONE-HOT ENCODING
# ─────────────────────────────────────────────

def one_hot_encode(labels, num_classes=NUM_CLASSES):
    N       = len(labels)
    one_hot = np.zeros((N, num_classes), dtype=np.float32)
    one_hot[np.arange(N), labels] = 1.0
    return one_hot


# ─────────────────────────────────────────────
# 4. NORMALIZE
# ─────────────────────────────────────────────

def normalize(images):
    """Normalize to [0, 1]. Safe to call even if already normalized."""
    if images.max() > 1.0:
        return images.astype(np.float32) / 255.0
    return images.astype(np.float32)


# ─────────────────────────────────────────────
# 5. DATA AUGMENTATION
# ─────────────────────────────────────────────

def augment_images(images, labels, seed=42):
    """
    Augment training set with:
      - Horizontal flip
      - Vertical flip
      - Brightness adjustment
      - 90-degree rotation

    Returns original + augmented (4x total).
    Only applied to tomato disease images, NOT to non_tomato class,
    to keep the rejection class realistic.
    """
    np.random.seed(seed)
    aug_images = []
    aug_labels = []

    non_tomato_idx = TOMATO_CLASSES.index("non_tomato")

    for img, lbl in zip(images, labels):
        # Always keep original
        aug_images.append(img)
        aug_labels.append(lbl)

        # Skip augmentation for non_tomato — keep it as real-world as possible
        if lbl == non_tomato_idx:
            continue

        # Horizontal flip
        aug_images.append(img[:, ::-1, :])
        aug_labels.append(lbl)

        # Vertical flip
        aug_images.append(img[::-1, :, :])
        aug_labels.append(lbl)

        # Brightness
        factor = np.random.uniform(0.7, 1.3)
        aug_images.append(np.clip(img * factor, 0.0, 1.0))
        aug_labels.append(lbl)

        # 90-degree rotation
        aug_images.append(np.rot90(img, k=1))
        aug_labels.append(lbl)

    aug_images = np.array(aug_images, dtype=np.float32)
    aug_labels = np.array(aug_labels, dtype=np.int32)

    print(f"\n[Augmentation] Original : {len(images)}")
    print(f"[Augmentation] Augmented: {len(aug_images)}")

    return aug_images, aug_labels


# ─────────────────────────────────────────────
# 6. BATCH GENERATOR
# ─────────────────────────────────────────────

def get_batches(images, labels, batch_size=32):
    N = len(images)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        yield images[start:end], labels[start:end]


# ─────────────────────────────────────────────
# 7. FULL PIPELINE
# ─────────────────────────────────────────────

def preprocess(dataset_dir, image_size=(128, 128), max_per_class=None):
    """
    Full preprocessing pipeline.

    Returns dict with all splits and metadata.
    """
    # Step 1 — Load
    print("\n[Pipeline] Step 1: Loading images...")
    images, labels, classes = load_dataset(
        dataset_dir,
        image_size=image_size,
        max_per_class=max_per_class
    )

    # Step 2 — Normalize
    print("\n[Pipeline] Step 2: Normalizing...")
    images = normalize(images)

    # Step 3 — Shuffle
    print("\n[Pipeline] Step 3: Shuffling...")
    images, labels = shuffle_dataset(images, labels)

    # Step 4 — Split
    print("\n[Pipeline] Step 4: Splitting...")
    (train_X, train_y,
     val_X,   val_y,
     test_X,  test_y) = split_dataset(images, labels)

    # Step 5 — Augment training only
    print("\n[Pipeline] Step 5: Augmenting training set...")
    train_X, train_y = augment_images(train_X, train_y)

    # Step 6 — One-hot encode
    print("\n[Pipeline] Step 6: One-hot encoding...")
    train_y_oh = one_hot_encode(train_y)
    val_y_oh   = one_hot_encode(val_y)
    test_y_oh  = one_hot_encode(test_y)

    print("\n[Pipeline] Preprocessing complete!")
    print(f"  Train : {train_X.shape} | Labels: {train_y_oh.shape}")
    print(f"  Val   : {val_X.shape}   | Labels: {val_y_oh.shape}")
    print(f"  Test  : {test_X.shape}  | Labels: {test_y_oh.shape}")

    return {
        "train_X"    : train_X,
        "train_y"    : train_y_oh,
        "val_X"      : val_X,
        "val_y"      : val_y_oh,
        "test_X"     : test_X,
        "test_y"     : test_y_oh,
        "classes"    : classes,
        "num_classes": NUM_CLASSES
    }


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  MODULE 2: PREPROCESSING — Quick Test")
    print("=" * 55)

    data = preprocess("dataset", image_size=(128, 128), max_per_class=10)

    print("\n[Test] Batch generator...")
    for bx, by in get_batches(data["train_X"], data["train_y"], batch_size=32):
        pass
    print(f"  Last batch X : {bx.shape}")
    print(f"  Last batch y : {by.shape}")
    print("\n  MODULE 2 TEST PASSED ✓")