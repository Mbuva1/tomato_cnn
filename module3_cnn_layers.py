"""
=============================================================
MODULE 3: CNN LAYERS
Project: Tomato Leaf Disease Detection using CNN from Scratch
Tools: Pure Python + NumPy only
=============================================================

Layers:
  1. Convolutional Layer
  2. ReLU Activation
  3. Max Pooling Layer
  4. Flatten Layer
  5. Fully Connected (Dense) Layer
  6. Dropout Layer
  7. Batch Normalization Layer
  8. Softmax Output
=============================================================
"""

import numpy as np


# ─────────────────────────────────────────────
# 1. CONVOLUTIONAL LAYER
# ─────────────────────────────────────────────

class ConvLayer:
    def __init__(self, num_filters, filter_size, stride=1, padding='same'):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride      = stride
        self.padding     = padding
        self.filters     = None
        self.biases      = None
        self.initialized = False
        self.cache       = {}

    def _initialize(self, input_channels):
        scale = np.sqrt(2.0 / (input_channels * self.filter_size ** 2))
        self.filters = np.random.randn(
            self.num_filters, self.filter_size, self.filter_size, input_channels
        ).astype(np.float32) * scale
        self.biases  = np.zeros(self.num_filters, dtype=np.float32)
        self.initialized = True

    def _pad(self, X):
        if self.padding == 'same':
            p = self.filter_size // 2
            return np.pad(X, ((0,0),(p,p),(p,p),(0,0)), mode='constant')
        return X

    def forward(self, X):
        if not self.initialized:
            self._initialize(X.shape[3])

        batch, H, W, C = X.shape
        F, S           = self.filter_size, self.stride
        X_pad          = self._pad(X)
        out_H          = (X_pad.shape[1] - F) // S + 1
        out_W          = (X_pad.shape[2] - F) // S + 1
        output         = np.zeros((batch, out_H, out_W, self.num_filters), dtype=np.float32)

        filters_flat = self.filters.reshape(self.num_filters, -1)
        for i in range(out_H):
            for j in range(out_W):
                patch = X_pad[:, i*S:i*S+F, j*S:j*S+F, :]
                output[:, i, j, :] = patch.reshape(batch, -1) @ filters_flat.T + self.biases

        self.cache['X']       = X
        self.cache['X_padded'] = X_pad
        return output

    def backward(self, dout):
        X        = self.cache['X']
        X_pad    = self.cache['X_padded']
        F, S     = self.filter_size, self.stride
        _, out_H, out_W, _ = dout.shape

        dX_pad        = np.zeros_like(X_pad)
        self.dfilters = np.zeros_like(self.filters)
        self.dbiases  = np.sum(dout, axis=(0, 1, 2))

        for i in range(out_H):
            for j in range(out_W):
                patch = X_pad[:, i*S:i*S+F, j*S:j*S+F, :]
                d     = dout[:, i, j, :]
                for f in range(self.num_filters):
                    self.dfilters[f] += np.sum(patch * d[:, f].reshape(-1,1,1,1), axis=0)
                    dX_pad[:, i*S:i*S+F, j*S:j*S+F, :] += \
                        self.filters[f] * d[:, f].reshape(-1,1,1,1)

        if self.padding == 'same':
            p  = F // 2
            dX = dX_pad[:, p:-p, p:-p, :]
        else:
            dX = dX_pad
        return dX


# ─────────────────────────────────────────────
# 2. RELU
# ─────────────────────────────────────────────

class ReLU:
    def __init__(self):
        self.cache = {}

    def forward(self, X):
        self.cache['X'] = X
        return np.maximum(0, X)

    def backward(self, dout):
        return dout * (self.cache['X'] > 0).astype(np.float32)


# ─────────────────────────────────────────────
# 3. MAX POOLING
# ─────────────────────────────────────────────

class MaxPoolLayer:
    def __init__(self, pool_size=2, stride=None):
        self.pool_size = pool_size
        self.stride    = stride if stride else pool_size
        self.cache     = {}

    def forward(self, X):
        batch, H, W, C = X.shape
        P, S           = self.pool_size, self.stride
        out_H          = (H - P) // S + 1
        out_W          = (W - P) // S + 1
        output         = np.zeros((batch, out_H, out_W, C), dtype=np.float32)

        for i in range(out_H):
            for j in range(out_W):
                patch = X[:, i*S:i*S+P, j*S:j*S+P, :]
                output[:, i, j, :] = np.max(patch, axis=(1, 2))

        self.cache['X']     = X
        self.cache['out_H'] = out_H
        self.cache['out_W'] = out_W
        return output

    def backward(self, dout):
        X             = self.cache['X']
        batch, H, W, C = X.shape
        P, S          = self.pool_size, self.stride
        out_H         = self.cache['out_H']
        out_W         = self.cache['out_W']
        dX            = np.zeros_like(X)

        for i in range(out_H):
            for j in range(out_W):
                patch    = X[:, i*S:i*S+P, j*S:j*S+P, :]
                max_vals = np.max(patch, axis=(1,2), keepdims=True)
                mask     = (patch == max_vals).astype(np.float32)
                dX[:, i*S:i*S+P, j*S:j*S+P, :] += \
                    mask * dout[:, i, j, :].reshape(batch, 1, 1, C)
        return dX


# ─────────────────────────────────────────────
# 4. FLATTEN
# ─────────────────────────────────────────────

class FlattenLayer:
    def __init__(self):
        self.cache = {}

    def forward(self, X):
        self.cache['shape'] = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.cache['shape'])


# ─────────────────────────────────────────────
# 5. DENSE LAYER
# ─────────────────────────────────────────────

class DenseLayer:
    def __init__(self, input_size, output_size):
        scale        = np.sqrt(2.0 / input_size)
        self.weights = np.random.randn(input_size, output_size).astype(np.float32) * scale
        self.biases  = np.zeros(output_size, dtype=np.float32)
        self.cache   = {}

    def forward(self, X):
        self.cache['X'] = X
        return X @ self.weights + self.biases

    def backward(self, dout):
        X              = self.cache['X']
        self.dweights  = X.T @ dout
        self.dbiases   = np.sum(dout, axis=0)
        return dout @ self.weights.T


# ─────────────────────────────────────────────
# 6. DROPOUT
# ─────────────────────────────────────────────

class Dropout:
    """
    Dropout layer — randomly zeroes neurons during training.
    Reduces overfitting. Disabled during inference.

    Args:
        rate : fraction of neurons to drop (e.g. 0.5 = drop 50%)
    """
    def __init__(self, rate=0.5):
        self.rate    = rate
        self.cache   = {}
        self.training = True   # set to False during prediction

    def forward(self, X):
        if not self.training:
            return X
        mask = (np.random.rand(*X.shape) > self.rate).astype(np.float32)
        self.cache['mask'] = mask
        return X * mask / (1.0 - self.rate)   # inverted dropout

    def backward(self, dout):
        if not self.training:
            return dout
        return dout * self.cache['mask'] / (1.0 - self.rate)


# ─────────────────────────────────────────────
# 7. BATCH NORMALIZATION
# ─────────────────────────────────────────────

class BatchNorm:
    """
    Batch Normalization — normalizes layer inputs to stabilize training.
    Improves accuracy and speeds up convergence.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.eps         = eps
        self.momentum    = momentum
        self.gamma       = np.ones(num_features,  dtype=np.float32)
        self.beta        = np.zeros(num_features, dtype=np.float32)
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var  = np.ones(num_features,  dtype=np.float32)
        self.cache        = {}
        self.training     = True

    def forward(self, X):
        if self.training:
            mean = np.mean(X, axis=0)
            var  = np.var(X,  axis=0)
            X_norm = (X - mean) / np.sqrt(var + self.eps)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var  = self.momentum * self.running_var  + (1 - self.momentum) * var
            self.cache        = {'X': X, 'mean': mean, 'var': var, 'X_norm': X_norm}
        else:
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * X_norm + self.beta

    def backward(self, dout):
        X      = self.cache['X']
        mean   = self.cache['mean']
        var    = self.cache['var']
        X_norm = self.cache['X_norm']
        N      = X.shape[0]

        dgamma = np.sum(dout * X_norm, axis=0)
        dbeta  = np.sum(dout, axis=0)
        self.dgamma = dgamma
        self.dbeta  = dbeta

        dX_norm = dout * self.gamma
        dvar    = np.sum(dX_norm * (X - mean) * -0.5 * (var + self.eps) ** -1.5, axis=0)
        dmean   = np.sum(dX_norm * -1 / np.sqrt(var + self.eps), axis=0) + \
                  dvar * np.mean(-2 * (X - mean), axis=0)
        dX      = dX_norm / np.sqrt(var + self.eps) + \
                  dvar * 2 * (X - mean) / N + dmean / N
        return dX


# ─────────────────────────────────────────────
# 8. SOFTMAX
# ─────────────────────────────────────────────

class Softmax:
    def __init__(self):
        self.cache = {}

    def forward(self, X):
        exp_x = np.exp(X - np.max(X, axis=1, keepdims=True))
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.cache['probs'] = probs
        return probs

    def backward(self, dout):
        return dout


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  MODULE 3: CNN LAYERS — Quick Test")
    print("=" * 55)

    np.random.seed(42)
    batch = np.random.rand(4, 128, 128, 3).astype(np.float32)
    print(f"\n[Input] {batch.shape}")

    out = ConvLayer(16, 3).forward(batch);    print(f"[Conv1]   {out.shape}")
    out = ReLU().forward(out);                print(f"[ReLU]    {out.shape}")
    out = MaxPoolLayer(2).forward(out);       print(f"[Pool]    {out.shape}")
    out = FlattenLayer().forward(out);        print(f"[Flatten] {out.shape}")
    out = DenseLayer(out.shape[1], 256).forward(out); print(f"[Dense]   {out.shape}")
    out = Dropout(0.5).forward(out);          print(f"[Dropout] {out.shape}")
    out = DenseLayer(256, 11).forward(out);   print(f"[Dense2]  {out.shape}")
    out = Softmax().forward(out);             print(f"[Softmax] {out.shape}")
    print(f"[Softmax] Probs sum: {out.sum(axis=1)}")
    print("\n  MODULE 3 TEST PASSED ✓")