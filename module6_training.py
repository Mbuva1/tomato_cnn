"""
=============================================================
MODULE 6: TRAINING LOOP
Project: Tomato Leaf Disease Detection using CNN from Scratch
Tools: Pure Python + NumPy only
=============================================================

Usage:
  Full training from scratch:
    python module6_training.py full

  Resume from saved weights (continue from epoch 7):
    python module6_training.py resume
=============================================================
"""

import numpy as np
import os
import time
from module2_preprocessing import preprocess, get_batches
from module4_forward_pass import TomatoCNN
from module5_backpropagation import SGDMomentum, backward_pass


# ─────────────────────────────────────────────
# PROGRESS BAR
# ─────────────────────────────────────────────

def progress_bar(batch, total, loss, acc, bar_len=25):
    filled = int(bar_len * batch / max(total, 1))
    bar    = '█' * filled + '░' * (bar_len - filled)
    print(f"\r  [{bar}] {batch}/{total} | Loss: {loss:.4f} | Acc: {acc*100:.1f}%", end='')


# ─────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────

def train_one_epoch(model, optimizer, train_X, train_y, batch_size=32):
    model.set_training(True)

    idx     = np.random.permutation(len(train_X))
    train_X = train_X[idx]
    train_y = train_y[idx]

    total_loss    = 0.0
    total_acc     = 0.0
    num_batches   = 0
    total_batches = int(np.ceil(len(train_X) / batch_size))

    for batch_X, batch_y in get_batches(train_X, train_y, batch_size):
        probs = model.forward(batch_X)
        loss  = model.compute_loss(probs, batch_y)
        acc   = model.compute_accuracy(probs, batch_y)
        backward_pass(model, probs, batch_y, optimizer)

        total_loss  += loss
        total_acc   += acc
        num_batches += 1
        progress_bar(num_batches, total_batches, loss, acc)

    print()
    return total_loss / num_batches, total_acc / num_batches


# ─────────────────────────────────────────────
# VALIDATE
# ─────────────────────────────────────────────

def validate(model, val_X, val_y, batch_size=32):
    model.set_training(False)

    total_loss  = 0.0
    total_acc   = 0.0
    num_batches = 0

    for batch_X, batch_y in get_batches(val_X, val_y, batch_size):
        probs = model.forward(batch_X)
        total_loss += model.compute_loss(probs, batch_y)
        total_acc  += model.compute_accuracy(probs, batch_y)
        num_batches += 1

    model.set_training(True)
    return total_loss / num_batches, total_acc / num_batches


# ─────────────────────────────────────────────
# SAVE HISTORY
# ─────────────────────────────────────────────

def save_history(history, filepath="saved_weights/history.csv"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")
        for row in history:
            f.write(",".join(str(x) for x in row) + "\n")


# ─────────────────────────────────────────────
# LOAD HISTORY (for resume)
# ─────────────────────────────────────────────

def load_history(filepath="saved_weights/history.csv"):
    """Load existing history so resume continues from correct epoch number."""
    history = []
    best_val_acc = 0.0
    if not os.path.exists(filepath):
        return history, best_val_acc
    with open(filepath, 'r') as f:
        lines = f.readlines()[1:]   # skip header
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 6:
                history.append([
                    int(parts[0]),
                    float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4]),
                    float(parts[5])
                ])
                best_val_acc = max(best_val_acc, float(parts[4]))
    return history, best_val_acc


# ─────────────────────────────────────────────
# FULL TRAINING LOOP
# ─────────────────────────────────────────────

def train(
    dataset_dir   = "dataset",
    image_size    = (64, 64),
    max_per_class = None,
    epochs        = 50,
    batch_size    = 32,
    learning_rate = 0.001,
    momentum      = 0.9,
    decay         = 0.0005,
    save_dir      = "saved_weights",
    patience      = 50,
    resume        = False       # ← set True to continue from saved weights
):
    print("=" * 55)
    print("  TOMATO CNN — TRAINING")
    print("=" * 55)
    print(f"  Mode          : {'RESUME' if resume else 'FRESH START'}")
    print(f"  Epochs        : {epochs}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Learning rate : {learning_rate}  (decay={decay})")
    print(f"  Momentum      : {momentum}")
    print(f"  Images/class  : {'ALL' if max_per_class is None else max_per_class}")
    print(f"  Image size    : {image_size}")
    print(f"  Early stop    : patience={patience}")
    print("=" * 55)

    # ── Preprocess ──
    data = preprocess(
        dataset_dir   = dataset_dir,
        image_size    = image_size,
        max_per_class = max_per_class
    )

    train_X     = data['train_X']
    train_y     = data['train_y']
    val_X       = data['val_X']
    val_y       = data['val_y']
    num_classes = data['num_classes']

    # ── Build model ──
    print("\n[Training] Building model...")
    model     = TomatoCNN(num_classes=num_classes)
    optimizer = SGDMomentum(
        learning_rate = learning_rate,
        momentum      = momentum,
        decay         = decay
    )

    os.makedirs(save_dir, exist_ok=True)

    # ── RESUME: load weights + history ──
    start_epoch  = 1
    history      = []
    best_val_acc = 0.0
    patience_count = 0

    weights_path = os.path.join(save_dir, "best_model.npz")

    if resume:
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            history, best_val_acc = load_history(
                os.path.join(save_dir, "history.csv")
            )
            start_epoch = len(history) + 1
            print(f"\n[Resume] Loaded weights from {weights_path}")
            print(f"[Resume] Continuing from epoch {start_epoch}")
            print(f"[Resume] Best val acc so far: {best_val_acc*100:.2f}%")
        else:
            print(f"\n[Resume] No saved weights found at {weights_path}")
            print(f"[Resume] Starting fresh instead.")

    print(f"\n[Training] Train : {len(train_X)} images")
    print(f"[Training] Val   : {len(val_X)} images")
    print(f"[Training] Starting from epoch {start_epoch}/{epochs}\n")

    for epoch in range(start_epoch, epochs + 1):
        start = time.time()

        print(f"Epoch {epoch}/{epochs}  (LR={optimizer.lr:.6f})")
        print("-" * 55)

        train_loss, train_acc = train_one_epoch(
            model, optimizer, train_X, train_y, batch_size
        )
        val_loss, val_acc = validate(model, val_X, val_y, batch_size)

        optimizer.step_epoch()
        elapsed = time.time() - start

        print(f"  Train → Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
        print(f"  Val   → Loss: {val_loss:.4f}   | Acc: {val_acc*100:.2f}%")
        print(f"  Time  : {elapsed:.1f}s")

        history.append([
            epoch,
            round(train_loss, 6), round(train_acc, 6),
            round(val_loss,   6), round(val_acc,   6),
            round(optimizer.lr, 8)
        ])

        # ── Save best model ──
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            model.save_weights(weights_path)
            print(f"  ★ Best model saved! Val Acc: {val_acc*100:.2f}%")
        else:
            patience_count += 1
            print(f"  No improvement ({patience_count}/{patience})")

        # ── Checkpoint every 5 epochs ──
        if epoch % 5 == 0:
            model.save_weights(os.path.join(save_dir, f"checkpoint_epoch{epoch}.npz"))
            print(f"  [Checkpoint] epoch {epoch} saved")

        # ── Save history after every epoch ──
        save_history(history)

        # ── Early stopping ──
        if patience_count >= patience:
            print(f"\n[Early Stop] No improvement for {patience} epochs. Stopping.")
            break

    print("\n" + "=" * 55)
    print(f"  TRAINING COMPLETE!")
    print(f"  Best Val Accuracy : {best_val_acc*100:.2f}%")
    print(f"  Weights saved to  : {save_dir}/")
    print("=" * 55)

    return model, history


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    if mode == "full":
        # Fresh training from scratch
        model, history = train(
            dataset_dir   = "dataset",
            image_size    = (64, 64),
            max_per_class = None,
            epochs        = 25,
            batch_size    = 32,
            learning_rate = 0.001,
            momentum      = 0.9,
            decay         = 0.0005,
            patience      = 25,
            resume        = False
        )

    elif mode == "resume":
        # Continue from saved weights — does NOT delete best_model.npz
        model, history = train(
            dataset_dir   = "dataset",
            image_size    = (64, 64),
            max_per_class = None,
            epochs        = 25,
            batch_size    = 32,
            learning_rate = 0.001,
            momentum      = 0.9,
            decay         = 0.0005,
            patience      = 25,
            resume        = True    # ← loads saved weights and continues
        )

    else:
        # Quick test
        print("Quick test (3 epochs, 5 images/class)...")
        print("Commands:")
        print("  Full training : python module6_training.py full")
        print("  Resume        : python module6_training.py resume\n")
        model, history = train(
            dataset_dir   = "dataset",
            image_size    = (64, 64),
            max_per_class = 5,
            epochs        = 3,
            batch_size    = 8,
            learning_rate = 0.001,
            momentum      = 0.9,
            decay         = 0.0005,
            patience      = 25,
            resume        = False
        )

    print("\n  MODULE 6 PASSED ✓")