"""
=============================================================
MODULE 5: BACKPROPAGATION
Project: Tomato Leaf Disease Detection using CNN from Scratch
Tools: Pure Python + NumPy only
=============================================================

Optimizer: SGD with Momentum + Learning Rate Decay
=============================================================
"""

import numpy as np
from module4_forward_pass import TomatoCNN


# ─────────────────────────────────────────────
# 1. LOSS GRADIENT
# ─────────────────────────────────────────────

def compute_loss_gradient(probs, labels_onehot):
    """Gradient of Cross-Entropy + Softmax combined."""
    return (probs - labels_onehot) / len(probs)


# ─────────────────────────────────────────────
# 2. SGD WITH MOMENTUM + LR DECAY
# ─────────────────────────────────────────────

class SGDMomentum:
    """
    SGD with Momentum and optional learning rate decay.

    Args:
        learning_rate : Initial learning rate
        momentum      : Momentum factor (0.9 recommended)
        decay         : LR multiplied by (1 - decay) each epoch
    """

    def __init__(self, learning_rate=0.01, momentum=0.9, decay=0.0):
        self.lr       = learning_rate
        self.init_lr  = learning_rate
        self.momentum = momentum
        self.decay    = decay
        self.velocity = {}
        self.epoch    = 0

    def step_epoch(self):
        """Call once per epoch to apply learning rate decay."""
        self.epoch += 1
        if self.decay > 0:
            self.lr = self.init_lr * (1.0 / (1.0 + self.decay * self.epoch))

    def update(self, name, param, grad):
        if name not in self.velocity:
            self.velocity[name] = np.zeros_like(param)
        self.velocity[name] = self.momentum * self.velocity[name] - self.lr * grad
        return param + self.velocity[name]


# ─────────────────────────────────────────────
# 3. FULL BACKWARD PASS
# ─────────────────────────────────────────────

def backward_pass(model, probs, labels_onehot, optimizer):
    """
    Backpropagate through the full CNN and update all weights.

    Args:
        model         : TomatoCNN instance
        probs         : Output probabilities (batch, num_classes)
        labels_onehot : True labels one-hot  (batch, num_classes)
        optimizer     : SGDMomentum instance
    """

    # ── Loss gradient → Softmax ──
    dout = compute_loss_gradient(probs, labels_onehot)
    dout = model.softmax.backward(dout)

    # ── Dense2 ──
    dout = model.dense2.backward(dout)
    model.dense2.weights = optimizer.update('d2_w', model.dense2.weights, model.dense2.dweights)
    model.dense2.biases  = optimizer.update('d2_b', model.dense2.biases,  model.dense2.dbiases)

    # ── Dropout ──
    dout = model.drop1.backward(dout)

    # ── ReLU4 ──
    dout = model.relu4.backward(dout)

    # ── BatchNorm4 ──
    dout = model.bn4.backward(dout)
    model.bn4.gamma = optimizer.update('bn4_g', model.bn4.gamma, model.bn4.dgamma)
    model.bn4.beta  = optimizer.update('bn4_b', model.bn4.beta,  model.bn4.dbeta)

    # ── Dense1 ──
    dout = model.dense1.backward(dout)
    model.dense1.weights = optimizer.update('d1_w', model.dense1.weights, model.dense1.dweights)
    model.dense1.biases  = optimizer.update('d1_b', model.dense1.biases,  model.dense1.dbiases)

    # ── Flatten ──
    dout = model.flatten.backward(dout)

    # ── Pool3 ──
    dout = model.pool3.backward(dout)

    # ── ReLU3 ──
    dout = model.relu3.backward(dout)

    # ── BatchNorm3 ──
    dout = model.bn3.backward(dout)
    model.bn3.gamma = optimizer.update('bn3_g', model.bn3.gamma, model.bn3.dgamma)
    model.bn3.beta  = optimizer.update('bn3_b', model.bn3.beta,  model.bn3.dbeta)

    # ── Conv3 ──
    dout = model.conv3.backward(dout)
    model.conv3.filters = optimizer.update('c3_f', model.conv3.filters, model.conv3.dfilters)
    model.conv3.biases  = optimizer.update('c3_b', model.conv3.biases,  model.conv3.dbiases)

    # ── Pool2 ──
    dout = model.pool2.backward(dout)

    # ── ReLU2 ──
    dout = model.relu2.backward(dout)

    # ── BatchNorm2 ──
    dout = model.bn2.backward(dout)
    model.bn2.gamma = optimizer.update('bn2_g', model.bn2.gamma, model.bn2.dgamma)
    model.bn2.beta  = optimizer.update('bn2_b', model.bn2.beta,  model.bn2.dbeta)

    # ── Conv2 ──
    dout = model.conv2.backward(dout)
    model.conv2.filters = optimizer.update('c2_f', model.conv2.filters, model.conv2.dfilters)
    model.conv2.biases  = optimizer.update('c2_b', model.conv2.biases,  model.conv2.dbiases)

    # ── Pool1 ──
    dout = model.pool1.backward(dout)

    # ── ReLU1 ──
    dout = model.relu1.backward(dout)

    # ── BatchNorm1 ──
    dout = model.bn1.backward(dout)
    model.bn1.gamma = optimizer.update('bn1_g', model.bn1.gamma, model.bn1.dgamma)
    model.bn1.beta  = optimizer.update('bn1_b', model.bn1.beta,  model.bn1.dbeta)

    # ── Conv1 ──
    dout = model.conv1.backward(dout)
    model.conv1.filters = optimizer.update('c1_f', model.conv1.filters, model.conv1.dfilters)
    model.conv1.biases  = optimizer.update('c1_b', model.conv1.biases,  model.conv1.dbiases)


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  MODULE 5: BACKPROPAGATION — Quick Test")
    print("=" * 55)

    np.random.seed(42)
    model     = TomatoCNN(num_classes=11)
    optimizer = SGDMomentum(learning_rate=0.01, momentum=0.9, decay=0.001)

    X      = np.random.rand(2, 128, 128, 3).astype(np.float32)
    labels = np.zeros((2, 11), dtype=np.float32)
    labels[0][3] = 1
    labels[1][10] = 1

    # Before
    probs_before = model.forward(X)
    loss_before  = model.compute_loss(probs_before, labels)
    print(f"\n[Before] Loss: {loss_before:.4f}")

    # Train 5 steps
    for step in range(5):
        model.set_training(True)
        probs = model.forward(X)
        loss  = model.compute_loss(probs, labels)
        backward_pass(model, probs, labels, optimizer)
        print(f"  Step {step+1} | Loss: {loss:.4f}")

    # After
    model.set_training(False)
    probs_after = model.forward(X)
    loss_after  = model.compute_loss(probs_after, labels)
    print(f"\n[After]  Loss: {loss_after:.4f}")

    if loss_after < loss_before:
        print("  Loss decreased — Backprop working! ✓")
    else:
        print("  Warning: Loss did not decrease.")

    print("\n  MODULE 5 TEST PASSED ✓")