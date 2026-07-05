from sklearn.datasets import load_breast_cancer, load_iris
import numpy as np

from logistic_regression.logistic_regression import LogisticRegression

np.random.seed(42)

# --- binary classification ---
print("Binary classification (breast cancer)")
X, y = load_breast_cancer(return_X_y=True)

# shuffle and split
indices = np.random.permutation(len(X))
X, y = X[indices], y[indices]
split = int(len(X) * .80)

clf = LogisticRegression(X[:split], y[:split])
clf.train()
clf.test(X[split:], y[split:])

# predict on a single sample
print("\nSingle prediction")
pred = clf.predict(X[-1:])
proba = clf.predict_proba(X[-1:])
print(f"Prediction: {pred}, actual: {y[-1]}")
print(f"Probability shape: {proba.shape}")
assert pred.shape == (1,)
assert proba.shape == (1, 2)
assert np.allclose(proba.sum(axis=1), 1)

# --- with regularization ---
print("\nWith regularization")
clf_reg = LogisticRegression(X[:split], y[:split], _lambda=1.0)
clf_reg.train()
clf_reg.test(X[split:], y[split:])

# --- multiclass classification ---
print("\nMulticlass classification (iris)")
X, y = load_iris(return_X_y=True)

indices = np.random.permutation(len(X))
X, y = X[indices], y[indices]
split = int(len(X) * .80)

clf_multi = LogisticRegression(X[:split], y[:split])
clf_multi.train()
clf_multi.test(X[split:], y[split:])

print(f"Weight shape: {clf_multi.W.shape}")
assert clf_multi.W.shape == (X.shape[1] + 1, 3)

proba = clf_multi.predict_proba(X[split:])
assert proba.shape == (len(X) - split, 3)
assert np.allclose(proba.sum(axis=1), 1)

print("\nAll checks passed.")
