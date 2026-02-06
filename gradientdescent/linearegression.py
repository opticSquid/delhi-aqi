import math
import numpy as np

from gradientdescent.backwaardpass import correction
from gradientdescent.computegradient import compute_linear_gradients
from gradientdescent.forwardpass import predict
from gradientdescent.lossfunction import compute_rmse_loss

class LinearRegression:
    def __init__(self, X, y, learning_rate=0.01, n_iters=1000):
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.decay_rate = 0.005
    
    def train_const_lr(self):
        _,n = self.X.shape
        weights = np.random.rand(n).reshape(n,1)
        bias = np.random.rand()

        w_history = []
        b_history = []

        for i in range(self.n_iters):
            y_hat = predict(self.X,weights,bias)

            loss = compute_rmse_loss(self.y,y_hat)

            if i % 100 == 0:
                print(f"Iteration {i}, MSE: {loss:.4f}")
                w_history.append(weights[0,0])
                b_history.append(bias)

            dW, db = compute_linear_gradients(self.X,self.y,y_hat)

            weights,bias = correction(self.learning_rate,weights,dW,bias,db)
                
        return weights,bias,w_history, b_history

    def train_decreasing_lr(self):
        _,n = self.X.shape
        weights = np.random.rand(n).reshape(n,1)
        bias = np.random.rand()
        learning_rate = self.learning_rate
        w_history = []
        b_history = []

        for i in range(self.n_iters):
            y_hat = predict(self.X,weights,bias)
        
            loss = compute_rmse_loss(self.y,y_hat)

            if i % 100 == 0:
                print(f"Iteration {i}, MSE: {loss:.4f}, learning rate: {learning_rate:.4f}")
                w_history.append(weights[0,0])
                b_history.append(bias)

            dW, db = compute_linear_gradients(self.X,self.y,y_hat)

            weights,bias = correction(learning_rate,weights,dW,bias,db)
            learning_rate = self.exponentialdecay(i)

        return weights,bias,w_history, b_history
    
    def exponentialdecay(self,t):
        return self.learning_rate * (math.e ** (-1*self.decay_rate*t))