import math
import numpy as np

from common import (Input, Output, confusion_matrix, analyze_confusion_matrix, softmax, CE_loss)

class LogisticRegression:

    def __init__(self, features: Input, targets: Output, _lambda=0.):
        """
        features: n by d matrix of inputs
        targets:  vector of outputs with values [0, num_classes]
        _lambda:  L2 regularization hyperparameter
        """
        # initialize weights to random for gradient descent
        assert len(targets.shape) == 1
        c = len(np.unique(targets))
        assert c > 1, "Must be at least 2 classes"

        n, d = features.shape
        self.W = np.random.random(size=(d+1, c)) # bias column
        self._lambda = _lambda
        self.X_raw = features.copy() # for hyperparameter optimization

        # normalize features
        self.mu = features.mean(axis=0)
        self.sigma = features.std(axis=0)
        self.X = (features - self.mu) / self.sigma
        self.X = np.hstack([np.ones((n, 1)), self.X])
        self.Y = np.zeros((n, c))
        self.Y[np.arange(n), targets] = 1 # one-hot encode targets

    def train(
        self, epochs=1_000, tol=1e-6,
        learning_rate=0.01, batch_size=32
    ):
        """
        Trains the model to initialize weights
        epochs: Number of training epochs for gradient descent
        tol: Tolerance for gradient norm to stop gradient descent
        learning_rate: Idk what you want me to say here
        batch_size: Batch size for mini-batch GD. 1 for SGD and n for batch GD
        """
        X, Y = self.X, self.Y
        n, _ = X.shape

        def gradient(X, Y):
            # regularization term
            reg = self._lambda * self.W
            reg[0, :] = 0

            # use MSE instead of SSE so gradient scales with batch size
            P = softmax(X @ self.W)
            return (1/len(X)) * X.T @ (P - Y) + reg

        # use gradient descent
        for _ in range(epochs):
            for i in range(math.ceil(n / batch_size)):
                # extract the batch
                start = (i * batch_size) % n
                end = start + batch_size
                X_batch = X[start:end, :]
                Y_batch = Y[start:end]

                # update weights
                grad = gradient(X_batch, Y_batch)
                self.W -= learning_rate * grad

            # check if we're within tolerance
            grad = gradient(X, Y)
            if np.linalg.norm(grad) < tol:
                print(f"Converged after {_} epochs")
                break

    def test(self, X_test: Input, actual: Output, print_=True):
        """
        X_test: n by d test input matrix
        actual: n by k output matrix
        """
        prediction_proba = self.predict_proba(X_test)
        prediction = np.argmax(prediction_proba, axis=1)
        y_one_hot = np.zeros((len(actual), prediction_proba.shape[1]))
        y_one_hot[np.arange(len(actual)), actual] = 1
        if prediction.shape != actual.shape:
            actual = actual.reshape(-1, 1)

        ce_loss = CE_loss(y_one_hot, prediction_proba)
        conf_mat = confusion_matrix(actual, prediction)
        prec, recall, f1, acc, spec = analyze_confusion_matrix(conf_mat)
        if print_:
            print(f"Confusion matrix:")
            print(conf_mat)
            print(f"Cross-entropy loss: {ce_loss:.6f}")
            print(f"Accuracy: {acc:.6f}")
            print(f"Error: {1 - acc:.6f}")
            print(f"Precision: {prec:.6f}")
            print(f"Recall: {recall:.6f}")
            print(f"F1 score: {f1:.6f}")
            print(f"Specificity: {spec:.6f}")

        return acc

    def predict_proba(self, X: Input) -> Output:
        """
        X: n by d input matrix to predict
        Returns: n by c output matrix where out[i][c] = P(C = c | x_i)
        """
        # normalize input with training statistics first
        n, _ = X.shape
        X_norm = (X - self.mu) / self.sigma

        # add bias column
        X_norm = np.hstack([np.ones((n, 1)), X_norm])
        return softmax(X_norm @ self.W)

    def predict(self, X: Input) -> Output:
        """
        X: n by d input matrix to predict
        Returns: output vector labelling which class the input is from
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

