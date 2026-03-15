import os
import csv

import matplotlib.pyplot as plt
import numpy as np

from knn.knn import KNN

DATA_DIR = "data"

# load iris data
with open(f"{DATA_DIR}/iris/iris.csv", "r") as f:
    raw_data = list(csv.reader(f))[1:]

iris_X = np.array([[float(num) for num in row[:-1]] for row in raw_data])
iris_Y = np.array([row[-1] for row in raw_data]).reshape(-1, 1)

# load face data
face_X_list = []
face_Y_list = []
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

            face_X_list.append(first_half)
            face_Y_list.append(second_half)

face_X = np.array(face_X_list)
face_Y = np.array(face_Y_list)

# shuffle and split iris
indices = np.random.permutation(len(iris_X))
iris_X, iris_Y = iris_X[indices], iris_Y[indices]
split = int(len(iris_X) * .80)

knn = KNN(task="C", features=iris_X[:split], targets=iris_Y[:split], k=3)
# knn.optimize_k(X=iris_X, Y=iris_Y)
knn.test(X_test=iris_X[split:], actual=iris_Y[split:])

inp = iris_X[-1:]
predicted = knn.predict(inp)
print(predicted, iris_Y[-1])

# shuffle and split faces
indices = np.random.permutation(len(face_X))
face_X, face_Y = face_X[indices], face_Y[indices]
split = int(len(face_X) * .80)

knn = KNN(task="R", features=face_X[:split], targets=face_Y[:split], k=5)
# knn.optimize_k(X=face_X, Y=face_Y)
knn.test(X_test=face_X[split:], actual=face_Y[split:])

inp = face_X[-1:]
predicted = knn.predict(inp)[0].reshape((112, 46))
plt.imshow(np.concatenate([face_X[-1].reshape(112, 46), predicted], axis=1), cmap="gray")
plt.show()
