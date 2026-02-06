import numpy as np

def predict(X,W:np.ndarray,b:float) -> np.ndarray:
    # X * W + b // @ == * or dot multiplication
    y_hat = X @ W+b
    return y_hat