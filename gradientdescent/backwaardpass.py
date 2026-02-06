import numpy as np
from typing import Tuple

def correction(lr: float, W:np.ndarray, dw: float, b: float, db: float) -> Tuple[np.ndarray, float]:
    W -= lr * dw
    b -= lr * db
    return W,b
