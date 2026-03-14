import numpy as np

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


