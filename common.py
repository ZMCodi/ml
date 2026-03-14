import numpy as np

# type aliases
Input = np.ndarray
Output = np.ndarray

def inner_prod(p: np.ndarray, q: np.ndarray):
    return np.dot(p, q)

def norm(p: np.ndarray):
    return np.sqrt(inner_prod(p, p))

def cosine_similarity(p: np.ndarray, q: np.ndarray):
    return inner_prod(p, q) / (norm(p) * norm(q)) \
    if norm(p) * norm(q) != 0 else 0

def euclidean_dist(p: np.ndarray, q: np.ndarray):
    return np.sqrt(np.square(p - q).sum())

def cosine_dist(p: np.ndarray, q: np.ndarray):
    return 1 - cosine_similarity(p, q)

def RMSE(actual: np.ndarray, pred: np.ndarray):
    return np.sqrt(np.square(actual - pred).mean())

def MAE(actual: np.ndarray, pred: np.ndarray):
    return np.abs(actual - pred).mean()

def MAPE(actual: np.ndarray, pred: np.ndarray):
    return np.abs((actual - pred) / (actual)).mean()

def confusion_matrix(actual: np.ndarray, pred: np.ndarray):
    actual = actual.flatten()
    pred = pred.flatten()

    classes = np.unique(actual)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    mat = np.zeros(shape=(len(classes), len(classes))).astype(int)
    for i in range(len(actual)):
        a = actual[i]
        p = pred[i]
        mat[class_to_idx[a]][class_to_idx[p]] += 1

    return mat

def analyze_confusion_matrix(mat: np.ndarray):
    mat_sum = mat.sum()

    precisions = []
    recalls = []
    f1s = []
    accuracies = []
    specificities = []

    for i in range(len(mat)):
        tp = mat[i, i]
        fn = mat[i, :].sum() - tp
        fp = mat[:, i].sum() - tp
        tn = mat_sum - tp - fn - fp

        precisions.append(tp / (tp + fp) if tp + fp != 0 else 0)
        recalls.append(tp / (tp + fn))
        f1s.append((2 * precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
                    if precisions[i] + recalls[i] != 0 else 0)
        accuracies.append((tp + tn) / mat_sum)
        specificities.append(tn / (tn + fp))

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    accuracies = np.array(accuracies)
    specificities = np.array(specificities)

    return precisions.mean(), recalls.mean(), f1s.mean(), accuracies.mean(), specificities.mean()


