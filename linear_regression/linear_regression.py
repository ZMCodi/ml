import numpy as np

from common import (Input, Output, RMSE, MAE, MAPE)

class LinearRegression:

    def __init__(self, features: Input, targets: Output):
        # initialize weights to random for gradient descent
        # and add an extra column for bias
        n, d = features.shape
        k = targets.shape[1] if len(targets.shape) == 2 else 1
        self.W = np.random.random(size=(d+1, k))

        # normalize features
        self.mu = features.mean(axis=0)
        self.sigma = features.std(axis=0)
        self.X = (features - self.mu) / self.sigma
        self.X = np.hstack([np.ones((n, 1)), self.X])
        self.Y = targets

    def train(self, _lambda=0):
        """
        Trains the model to initialize weights
        _lambda: Optional L2 regularization constant
        """
        # gradient is X.T @ X @ W - X.T @ Y
        # use solve since its more numerically stable
        X, Y = self.X, self.Y
        self.W = np.linalg.solve(X.T @ X, X.T @ Y)

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
        X_norm = np.hstack([np.ones((n, 1)), X_norm])
        return X_norm @ self.W

# from sklearn.datasets import fetch_california_housing
# data = fetch_california_housing()
# from sklearn.datasets import make_regression
# X, y, true_coef = make_regression(n_samples=1000, n_features=3, noise=5, coef=True, random_state=42)
#
# lr = LinearRegression(X, y)
# lr.train()
#
# print(MAPE(lr.W[1:], true_coef))

import os
import matplotlib.pyplot as plt

DATA_DIR = "data"
# load face data
face_data: list[tuple[np.ndarray, np.ndarray]] = []
face_data_dir = DATA_DIR + "/att_faces"
for folder in os.listdir(face_data_dir):
    path = face_data_dir + "/" + folder
    if os.path.isdir(path):
        for img in os.listdir(path):
            face = plt.imread(path + "/" + img, "r")

            # split face into half
            cols = face.shape[-1]
            first_half = face[:, :cols // 2].flatten()
            second_half = face[:, cols // 2:].flatten()

            face_data.append((first_half, second_half))

np.random.shuffle(face_data)
X = np.array([x for x, _ in face_data])
Y = np.array([y for _, y in face_data])

lr = LinearRegression(X[:-1], Y[:-1])
lr.train()

inp = X[-1:]
lr.predict(inp)
pred = lr.predict(inp).reshape((112, 46))
plt.imshow(np.concatenate([inp.reshape(112,46), pred], axis=1), cmap="gray")
plt.show()
