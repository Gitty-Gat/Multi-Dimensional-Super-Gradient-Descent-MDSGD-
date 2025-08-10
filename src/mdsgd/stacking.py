import numpy as np
from typing import List


def logistic_stack(P: np.ndarray, y: np.ndarray, iters: int = 200, lr: float = 0.1):
    """
    P: shape (U, N) candidate probabilities; y: shape (N,)
    Returns weights w (U,), simplex constrained via softmax.
    """
    U, N = P.shape
    logits = np.zeros(U)  # start uniform
    for _ in range(iters):
        w = np.exp(logits)
        w = w / w.sum()
        pred = (w[:, None] * P).sum(axis=0)
        grad = np.array([(pred - y) @ (P[u] - pred) / N for u in range(U)])
        logits -= lr * grad
    w = np.exp(logits)
    return w / w.sum()
