import math
import numpy as np

from common import (Input, Output, RMSE, MAE, MAPE, norm)

class LinearRegression:

    def __init__(self, features: Input, targets: Output):
        # initialize weights to random for gradient descent
        n, d = features.shape
        k = targets.shape[1] if len(targets.shape) == 2 else 1
        self.W = np.random.random(size=(d+1, k)) # bias column

        # normalize features
        self.mu = features.mean(axis=0)
        self.sigma = features.std(axis=0)
        self.X = (features - self.mu) / self.sigma
        self.X = np.hstack([np.ones((n, 1)), self.X])
        self.Y = targets

    def train(
            self, _lambda=0, epochs=1_000, tol=1e-6,
            learning_rate=0.01, batch_size=32
    ):
        """
        Trains the model to initialize weights
        _lambda: Optional L2 regularization constant
        epochs: Number of training epochs for gradient descent
        tol: Tolerance for gradient norm to stop gradient descent
        learning_rate: Idk what you want me to say here
        batch_size: Batch size for mini-batch GD. 1 for SGD and n for batch GD
        """
        # use solve since its more numerically stable
        X, Y = self.X, self.Y
        n, d = X.shape

        if d < 10e4:
            # analytical solution should work fine here
            if n > d:
                # use solve since its more numerically stable
                # W = (X.T X)^-1 X.T Y
                self.W = np.linalg.solve(X.T @ X, X.T @ Y)
            else:
                # underdetermined system
                # W = X.T (X X.T)^-1 Y
                # solve for A = (X X.T)^-1 Y
                # then W = X.T A
                A = np.linalg.solve(X @ X.T, Y)
                self.W = X.T @ A

        else:
            # use gradient descent
            n = len(X)
            for _ in range(epochs):
                for i in range(math.ceil(n / batch_size)):
                    # extract the batch
                    start = (i * batch_size) % n
                    end = start + batch_size
                    X_batch = X[start:end, :]
                    Y_batch = Y[start:end]

                    grad = X_batch.T @ (X_batch @ self.W - Y_batch)
                    self.W -= learning_rate * grad

        # TODO: add L2 regularization

    def test(self, X_test: Input, actual: Output, print_=True):
        prediction = self.predict(X_test)

        rmse = RMSE(actual, prediction)
        if print_:
            print(f"RMSE: {rmse:.6f}")
            print(f"MAE: {MAE(actual, prediction):.6f}")
            print(f"MAPE: {MAPE(actual, prediction):.6f}")
        return -rmse

    def predict(self, X: Input) -> Output:
        # normalize input with training statistics first
        n, _ = X.shape
        X_norm = (X - self.mu) / self.sigma

        # add bias column
        X_norm = np.hstack([np.ones((n, 1)), X_norm])
        return X_norm @ self.W

    def optimize_lambda(self):
        # TODO: implement this
        pass

# from sklearn.datasets import fetch_california_housing
# data = fetch_california_housing()
from sklearn.datasets import make_regression
X, y, true_coef = make_regression(n_samples=1000, n_features=3, noise=5, coef=True, random_state=42)

lr = LinearRegression(X, y)
lr.train()

print(MAPE(lr.W[1:], true_coef))

