import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x, dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        # Forward pass
        z1 = W1 @ x + b1
        a1 = np.maximum(0, z1)
        z2 = W2 @ a1 + b2

        # MSE loss
        n = len(y_true)
        loss = np.mean((z2 - y_true) ** 2)

        # Backward pass

        # dL/dz2
        dz2 = 2 * (z2 - y_true) / n

        # Layer 2 gradients
        # Outer product: output_size x hidden_size
        dW2 = np.outer(dz2, a1)
        db2 = dz2

        # Gradient through the second linear layer
        da1 = W2.T @ dz2

        # Gradient through ReLU
        relu_mask = (z1 > 0).astype(float)
        dz1 = da1 * relu_mask

        # Layer 1 gradients
        # Outer product: hidden_size x input_size
        dW1 = np.outer(dz1, x)
        db1 = dz1

        # Round only the final returned values
        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }