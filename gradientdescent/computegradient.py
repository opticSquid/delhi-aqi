import numpy as np
from typing import Tuple

def compute_linear_gradients(X: np.ndarray, y: np.ndarray, y_hat: np.ndarray)-> Tuple[float, float]:
    m = X.shape[0]
    diff = y_hat-y
    dW = (2/m) * (X.T @ diff)
    db = (2/m) * np.sum(diff)
    return dW, db