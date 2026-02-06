import numpy as np

def compute_rmse_loss(y:np.ndarray,y_hat:np.ndarray) -> np.floating:
    return np.sqrt(np.mean((y_hat-y)**2))