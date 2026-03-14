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
        data: list[tuple[Input, Output]],
        k: int
    ):
        """
        task: "R" for regression, "C" for classification
        data: List of (input, output) pairs
        k: number of neighbors to consider
        """
        self.data = data
        self.k = k
        self.task = task

    def test(self, test_data: list[tuple[Input, Output]], print_=True):
        """
        test_data: list of (input, actual output) pairs to test on the model
        """
        ins = [x for x, _ in test_data]
        actual = np.array([y for _, y in test_data])
        predictions = np.array(self.predict(ins))

        actual = np.array(actual)
        prediction = np.array(predictions)

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

    def predict(self, x: list[Input], distance_f: Callable = euclidean_dist) -> list[Output]:
        """
        x: list of input vectors to predict
        Returns: list of output vectors
        """
        output: list[Output] = []

        for inp in x:
            # max heap of (distance, index (tiebreaker), output)
            heap: list[tuple[float, int, np.ndarray]] = []

            for i, (dx, dy) in enumerate(self.data):
                dist = distance_f(inp, dx)

                # only store the k nearest neighbours
                if len(heap) < self.k:
                    heapq.heappush_max(heap, (dist, i, dy))
                elif dist < heap[0][0]:
                    heapq.heappop_max(heap)
                    heapq.heappush_max(heap, (dist, i, dy))

            # now we have the output of k nearest neighbours to x in heap
            outputs = [y for _, _, y in heap]

            if self.task == "R":
                # average over all neighbors
                output.append(np.array(outputs).mean(axis=0))

            elif self.task == "C":
                # find the most frequent class
                # y here should be singleton lists
                classes = np.array(outputs).flatten()
                mode = Counter(classes).most_common(1)[0][0]
                output.append(np.array([mode]))

        return output

    def optimize_k(self, k_fold=5, data_set: list[tuple[Input, Output]] | None = None):
        """
        Optimizes hyperparameter k by maximizing accuracy using k-fold CV
        k_fold: how many folds to split the data into
        data_set: optional data set to optimize on. If not provided, uses model's data
        """
        initial_data = self.data.copy()
        initial_k = self.k

        # split data into k_fold chunks
        full_data = data_set if data_set else self.data
        chunk_size = len(full_data) // k_fold
        chunks = [full_data[i:i+chunk_size] for i in range(0, len(full_data), chunk_size)]

        best_k = self.k
        best_acc = float("-inf")

        # try all odd k
        for k in range(1, chunk_size * (k_fold - 1), 2):
            accuracies = []
            for i, chunk in enumerate(chunks):

                # set i as test and rest as training
                test_set = chunk
                self.data = sum(chunks[:i] + chunks[i+1:], [])
                self.k = k

                accuracies.append(self.test(test_set, print_=False))

            # average accuracies and check if best
            acc = sum(accuracies) / len(accuracies)
            print(f"k: {k}, accuracy: {acc}")
            if acc > best_acc:
                best_acc = acc
                best_k = k

        print(f"Best k: {best_k} ({best_acc})")
        self.data = initial_data
        self.k = initial_k

