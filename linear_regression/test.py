from sklearn.datasets import make_regression
import numpy as np

from linear_regression.linear_regression import LinearRegression
from common import Input, Output, RMSE

np.random.seed(42)

# --- single output regression ---
X: Input
y: Output
X, y, true_coef = make_regression(n_samples=200, n_features=10, noise=5, random_state=42, coef=True)
split = 150

print("Overdetermined system")
lr = LinearRegression(X[:split], y[:split])
lr.train()
lr.test(X[split:], y[split:])

# predict on a single sample
print("\nSingle prediction")
pred = lr.predict(X[-1:])
print(f"Single prediction shape: {pred.shape}")
assert pred.shape == (1, 1)
print(f"RMSE with true coef: ", RMSE(lr.W[1:] / lr.sigma.reshape(-1, 1), true_coef.reshape(-1, 1)))

# --- with regularization ---
print("\nWith regularization")
lr_reg = LinearRegression(X[:split], y[:split], _lambda=1.0)
lr_reg.train()
lr_reg.test(X[split:], y[split:])

print(f"RMSE with true coef: ", RMSE(lr_reg.W[1:] / lr_reg.sigma.reshape(-1, 1), true_coef.reshape(-1, 1)))

# --- multioutput regression ---
print("\nMulti output")
X, y, true_coef = make_regression(n_samples=200, n_features=10, n_targets=3, noise=5, random_state=42, coef=True)

lr_multi = LinearRegression(X[:split], y[:split])
lr_multi.train()
lr_multi.test(X[split:], y[split:])
print(f"Multioutput weight shape: {lr_multi.W.shape}")
assert lr_multi.W.shape == (11, 3)
print(f"RMSE with true coef: ", RMSE(lr_multi.W[1:] / lr_multi.sigma.reshape(-1, 1), true_coef))

# --- gradient descent path (high-d to force GD branch) ---
print("\nGradient descent")
X, y, true_coef = make_regression(n_samples=200, n_features=15000, noise=5, random_state=42, coef=True)

lr_gd = LinearRegression(X[:split], y[:split])
lr_gd.train(epochs=500, learning_rate=.01, batch_size=split)
lr_gd.test(X[split:], y[split:])

print(f"RMSE with true coef: ", RMSE(lr_gd.W[1:] / lr_gd.sigma.reshape(-1, 1), true_coef.reshape(-1, 1)))

# --- underdetermined system (n < d) ---
print("\nUnderdetermined system")
X, y, true_coef = make_regression(n_samples=50, n_features=200, noise=5, random_state=42, coef=True)

lr_under = LinearRegression(X[:30], y[:30])
lr_under.train()
lr_under.test(X[30:], y[30:])
print(f"RMSE with true coef: ", RMSE(lr_under.W[1:] / lr_under.sigma.reshape(-1, 1), true_coef.reshape(-1, 1)))

# --- optimize_lambda ---
lr_cv = LinearRegression(X[:split], y[:split])
lr_cv.optimize_lambda(k_fold=5)

print("\nAll checks passed.")
