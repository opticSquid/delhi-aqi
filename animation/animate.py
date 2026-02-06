import numpy as np
import matplotlib.pyplot as plt

def plot_contour(X, y, w_history, b_history):
    # 1. Create a grid of weight and bias values
    w_range = np.linspace(min(w_history)-1, max(w_history)+1, 100)
    b_range = np.linspace(min(b_history)-1, max(b_history)+1, 100)
    W, B = np.meshgrid(w_range, b_range) #
    
    # 2. Calculate MSE loss for every point on the grid
    Z = np.array([np.mean((y - (X * w + b))**2) for w, b in zip(np.ravel(W), np.ravel(B))])
    Z = Z.reshape(W.shape)

    # 3. Plot the concentric circles (Contour)
    plt.figure(figsize=(10, 6))
    plt.contour(W, B, Z, levels=50, cmap='viridis') #
    
    # 4. Plot the "Moving Line" (Static Path)
    plt.plot(w_history, b_history, 'r-', label='Gradient Descent Path')
    plt.plot(w_history[-1], b_history[-1], 'go', label='Global Minimum') # Final point
    
    plt.xlabel('Weight (w)')
    plt.ylabel('Bias (b)')
    plt.title('Loss Surface Contour & Optimization Path')
    plt.legend()
    plt.show()