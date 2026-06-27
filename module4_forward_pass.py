"""
=============================================================
MODULE 4: FORWARD PASS & CNN MODEL
Project: Tomato Leaf Disease Detection using CNN from Scratch
Tools: Pure Python + NumPy only
=============================================================

Architecture (64x64 input):

  Input (64x64x3)
  → Conv1 (32 filters) → BN → ReLU → Pool → (32x32x32)
  → Conv2 (64 filters) → BN → ReLU → Pool → (16x16x64)
  → Conv3 (64 filters) → BN → ReLU → Pool → (8x8x64)
  → Flatten (4096)
  → Dense1 (256) → BN → ReLU → Dropout(0.5)
  → Dense2 (11)  → Softmax
=============================================================
"""

import os
import numpy as np
from module3_cnn_layers import (
    ConvLayer, ReLU, MaxPoolLayer,
    FlattenLayer, DenseLayer, Dropout,
    BatchNorm, Softmax
)


class TomatoCNN:
    """
    CNN for Tomato Leaf Disease Detection.
    11 output classes (10 diseases/healthy + 1 non_tomato).
    Input image size: 64x64x3
    """

    def __init__(self, num_classes=11):
        self.num_classes = num_classes

        # ── Block 1 ──
        self.conv1 = ConvLayer(32, filter_size=3, padding='same')
        self.bn1   = BatchNorm(32)
        self.relu1 = ReLU()
        self.pool1 = MaxPoolLayer(2)

        # ── Block 2 ──
        self.conv2 = ConvLayer(64, filter_size=3, padding='same')
        self.bn2   = BatchNorm(64)
        self.relu2 = ReLU()
        self.pool2 = MaxPoolLayer(2)

        # ── Block 3 ──
        self.conv3 = ConvLayer(64, filter_size=3, padding='same')
        self.bn3   = BatchNorm(64)
        self.relu3 = ReLU()
        self.pool3 = MaxPoolLayer(2)

        # ── Flatten ──
        self.flatten = FlattenLayer()

        # ── Dense 1 — 8*8*64 = 4096 ──
        self.dense1 = DenseLayer(4096, 256)
        self.bn4    = BatchNorm(256)
        self.relu4  = ReLU()
        self.drop1  = Dropout(rate=0.5)

        # ── Dense 2 (output) ──
        self.dense2  = DenseLayer(256, num_classes)
        self.softmax = Softmax()

        self.training = True

        print("[TomatoCNN] Model built!")
        print(f"  Input   : (batch, 64, 64, 3)")
        print(f"  Conv1   : (batch, 64, 64, 32)")
        print(f"  Pool1   : (batch, 32, 32, 32)")
        print(f"  Conv2   : (batch, 32, 32, 64)")
        print(f"  Pool2   : (batch, 16, 16, 64)")
        print(f"  Conv3   : (batch, 16, 16, 64)")
        print(f"  Pool3   : (batch,  8,  8, 64)")
        print(f"  Flatten : (batch, 4096)")
        print(f"  Dense1  : (batch, 256)")
        print(f"  Dense2  : (batch, {num_classes})")
        print(f"  Softmax : (batch, {num_classes})")


    # ─────────────────────────────────────────
    # TRAINING MODE
    # ─────────────────────────────────────────

    def set_training(self, mode: bool):
        self.training       = mode
        self.bn1.training   = mode
        self.bn2.training   = mode
        self.bn3.training   = mode
        self.bn4.training   = mode
        self.drop1.training = mode


    # ─────────────────────────────────────────
    # FORWARD PASS
    # ─────────────────────────────────────────

    def forward(self, X):
        out = self.conv1.forward(X)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)

        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)

        out = self.conv3.forward(out)
        out = self.bn3.forward(out)
        out = self.relu3.forward(out)
        out = self.pool3.forward(out)

        out = self.flatten.forward(out)

        out = self.dense1.forward(out)
        out = self.bn4.forward(out)
        out = self.relu4.forward(out)
        out = self.drop1.forward(out)

        out   = self.dense2.forward(out)
        probs = self.softmax.forward(out)

        return probs


    # ─────────────────────────────────────────
    # LOSS
    # ─────────────────────────────────────────

    def compute_loss(self, probs, labels_onehot):
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        return -np.sum(labels_onehot * np.log(probs_clipped)) / len(probs)


    # ─────────────────────────────────────────
    # ACCURACY
    # ─────────────────────────────────────────

    def compute_accuracy(self, probs, labels_onehot):
        return float(np.mean(
            np.argmax(probs, axis=1) == np.argmax(labels_onehot, axis=1)
        ))


    # ─────────────────────────────────────────
    # PREDICT
    # ─────────────────────────────────────────

    def predict(self, X, class_names=None):
        self.set_training(False)
        probs     = self.forward(X)
        self.set_training(True)
        class_ids = np.argmax(probs,  axis=1)
        confs     = np.max(probs,     axis=1)
        if class_names:
            return [(class_names[i], float(c)) for i, c in zip(class_ids, confs)]
        return [(int(i), float(c)) for i, c in zip(class_ids, confs)]


    # ─────────────────────────────────────────
    # SAVE WEIGHTS
    # ─────────────────────────────────────────

    def save_weights(self, filepath):
        if not filepath.endswith('.npz'):
            filepath += '.npz'
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        np.savez(filepath,
            conv1_filters  = self.conv1.filters,
            conv1_biases   = self.conv1.biases,
            conv2_filters  = self.conv2.filters,
            conv2_biases   = self.conv2.biases,
            conv3_filters  = self.conv3.filters,
            conv3_biases   = self.conv3.biases,
            dense1_weights = self.dense1.weights,
            dense1_biases  = self.dense1.biases,
            dense2_weights = self.dense2.weights,
            dense2_biases  = self.dense2.biases,
            bn1_gamma = self.bn1.gamma, bn1_beta = self.bn1.beta,
            bn1_rmean = self.bn1.running_mean, bn1_rvar = self.bn1.running_var,
            bn2_gamma = self.bn2.gamma, bn2_beta = self.bn2.beta,
            bn2_rmean = self.bn2.running_mean, bn2_rvar = self.bn2.running_var,
            bn3_gamma = self.bn3.gamma, bn3_beta = self.bn3.beta,
            bn3_rmean = self.bn3.running_mean, bn3_rvar = self.bn3.running_var,
            bn4_gamma = self.bn4.gamma, bn4_beta = self.bn4.beta,
            bn4_rmean = self.bn4.running_mean, bn4_rvar = self.bn4.running_var,
        )
        print(f"[Model] Weights saved → {filepath}")


    # ─────────────────────────────────────────
    # LOAD WEIGHTS
    # ─────────────────────────────────────────

    def load_weights(self, filepath):
        if not filepath.endswith('.npz'):
            filepath += '.npz'
        d = np.load(filepath)

        self.conv1.filters = d['conv1_filters']; self.conv1.biases = d['conv1_biases']; self.conv1.initialized = True
        self.conv2.filters = d['conv2_filters']; self.conv2.biases = d['conv2_biases']; self.conv2.initialized = True
        self.conv3.filters = d['conv3_filters']; self.conv3.biases = d['conv3_biases']; self.conv3.initialized = True
        self.dense1.weights = d['dense1_weights']; self.dense1.biases = d['dense1_biases']
        self.dense2.weights = d['dense2_weights']; self.dense2.biases = d['dense2_biases']

        self.bn1.gamma = d['bn1_gamma']; self.bn1.beta = d['bn1_beta']
        self.bn1.running_mean = d['bn1_rmean']; self.bn1.running_var = d['bn1_rvar']
        self.bn2.gamma = d['bn2_gamma']; self.bn2.beta = d['bn2_beta']
        self.bn2.running_mean = d['bn2_rmean']; self.bn2.running_var = d['bn2_rvar']
        self.bn3.gamma = d['bn3_gamma']; self.bn3.beta = d['bn3_beta']
        self.bn3.running_mean = d['bn3_rmean']; self.bn3.running_var = d['bn3_rvar']
        self.bn4.gamma = d['bn4_gamma']; self.bn4.beta = d['bn4_beta']
        self.bn4.running_mean = d['bn4_rmean']; self.bn4.running_var = d['bn4_rvar']

        print(f"[Model] Weights loaded ← {filepath}")


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  MODULE 4: FORWARD PASS — Quick Test")
    print("=" * 55)

    np.random.seed(42)
    model = TomatoCNN(num_classes=11)

    X      = np.random.rand(2, 64, 64, 3).astype(np.float32)
    labels = np.zeros((2, 11), dtype=np.float32)
    labels[0][2]  = 1
    labels[1][10] = 1   # non_tomato

    print(f"\n[Test] Input : {X.shape}")
    probs = model.forward(X)
    print(f"[Test] Output: {probs.shape}")
    print(f"[Test] Probs sum : {probs.sum(axis=1)}")
    print(f"[Test] Loss      : {model.compute_loss(probs, labels):.4f}")
    print(f"[Test] Accuracy  : {model.compute_accuracy(probs, labels)*100:.2f}%")

    print("\n  MODULE 4 TEST PASSED ✓")