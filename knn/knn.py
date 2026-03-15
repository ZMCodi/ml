from typing import Callable, Literal
from collections import Counter
import heapq

import numpy as np

from common import (RMSE, MAE, MAPE, euclidean_dist,
    confusion_matrix, analyze_confusion_matrix, Input, Output)

class KNN:
    def __init__(
        self,
        task: Literal["R", "C"],
        features: Input,
        targets: Output,
        k: int
    ):
        """
        task: "R" for regression, "C" for classification
        features: n by d matrix of inputs
        targets:  n by k matrix of outputs
        k: number of neighbors to consider
        """
        self.X = features.copy()
        self.Y = targets.reshape((len(targets), -1)).copy()
        self.k = k
        self.task = task

    def test(self, X_test: Input, actual: Output, print_=True):
        """
        X_test: n by d test input matrix
        actual: n by k output matrix
        """
        actual = actual.reshape((len(actual), -1))
        prediction = self.predict(X_test)

        if self.task == "R":
            # calculate RMSE, MAE, MAPE
            rmse = RMSE(actual, prediction)
            if print_:
                print(f"RMSE: {rmse:.6f}")
                print(f"MAE: {MAE(actual, prediction):.6f}")
                print(f"MAPE: {MAPE(actual, prediction):.6f}")
            return -rmse

        elif self.task == "C":
            # calculate accuracy, error
            conf_mat = confusion_matrix(actual, prediction)
            prec, recall, f1, acc, spec = analyze_confusion_matrix(conf_mat)
            if print_:
                print(f"Confusion matrix:")
                print(conf_mat)
                print(f"Accuracy: {acc:.6f}")
                print(f"Error: {1 - acc:.6f}")
                print(f"Precision: {prec:.6f}")
                print(f"Recall: {recall:.6f}")
                print(f"F1 score: {f1:.6f}")
                print(f"Specificity: {spec:.6f}")
            return f1

    def predict(self, X: Input, distance_f: Callable = euclidean_dist) -> Output:
        """
        X: n by d input matrix to predict
        Returns: n by k predicted output matrix
        """
        n = X.shape[0]
        k_out = self.Y.shape[1]
        output = np.empty((n, k_out), dtype=self.Y.dtype)

        for i in range(n):
            # max heap of (distance, index (tiebreaker), row index)
            heap: list[tuple[float, int, int]] = []

            for j in range(self.X.shape[0]):
                dist = distance_f(X[i], self.X[j])

                # only store the k nearest neighbours
                if len(heap) < self.k:
                    heapq.heappush_max(heap, (dist, j, j))
                elif dist < heap[0][0]:
                    heapq.heappop_max(heap)
                    heapq.heappush_max(heap, (dist, j, j))

            # now we have the k nearest neighbours
            neighbor_idx = [idx for _, _, idx in heap]
            neighbors = self.Y[neighbor_idx]

            if self.task == "R":
                # average over all neighbors
                output[i] = neighbors.mean(axis=0)

            elif self.task == "C":
                # find the most frequent class
                classes = neighbors.flatten()
                mode = Counter(classes).most_common(1)[0][0]
                output[i] = mode

        return output

    def optimize_k(self, k_fold=5, X: Input | None = None, Y: Output | None = None):
        """
        Optimizes hyperparameter k by maximizing accuracy using k-fold CV
        k_fold: how many folds to split the data into
        X, Y: optional data to optimize on. If not provided, uses model's data
        """
        X_full = X if X is not None else self.X
        Y_full = Y if Y is not None else self.Y
        Y_full = Y_full.reshape((len(Y_full), -1))

        # shuffle
        indices = np.random.permutation(len(X_full))
        X_full = X_full[indices]
        Y_full = Y_full[indices]

        initial_X, initial_Y = self.X.copy(), self.Y.copy()
        initial_k = self.k

        # split data into k_fold chunks
        chunk_size = len(X_full) // k_fold
        X_chunks = [X_full[i:i+chunk_size] for i in range(0, len(X_full), chunk_size)]
        Y_chunks = [Y_full[i:i+chunk_size] for i in range(0, len(Y_full), chunk_size)]

        best_k = self.k
        best_acc = float("-inf")

        # try all odd k
        for k in range(1, chunk_size * (k_fold - 1), 2):
            accuracies = []
            for i in range(len(X_chunks)):

                # set i as test and rest as training
                self.X = np.vstack(X_chunks[:i] + X_chunks[i+1:])
                self.Y = np.vstack(Y_chunks[:i] + Y_chunks[i+1:])
                self.k = k

                accuracies.append(self.test(X_chunks[i], Y_chunks[i], print_=False))

            # average accuracies and check if best
            acc = sum(accuracies) / len(accuracies)
            print(f"k: {k}, accuracy: {acc}")
            if acc > best_acc:
                best_acc = acc
                best_k = k

        print(f"Best k: {best_k} ({best_acc})")
        self.X = initial_X
        self.Y = initial_Y
        self.k = initial_k
