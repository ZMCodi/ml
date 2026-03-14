import os
import csv

import matplotlib.pyplot as plt
import numpy as np

from knn.knn import KNN

DATA_DIR = "data"

# load iris data
iris_data: list[tuple[np.ndarray, np.ndarray]] = []
with open(f"{DATA_DIR}/iris/iris.csv", "r") as f:
    raw_data = list(csv.reader(f))[1:]

    # convert to float
    for row in raw_data:
        inp = np.array([float(num) for num in row[:-1]])
        out = np.array([row[-1]])
        iris_data.append((inp, out))

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


np.random.shuffle(iris_data)
knn = KNN(task="C", data=iris_data[:int(len(iris_data) * .80)], k=3)
# knn.optimize_k(data_set=iris_data)
knn.test(test_data=iris_data[int(len(iris_data) * .80):])

inp = iris_data[-1][0]
predicted = knn.predict([inp])
print(predicted, iris_data[-1][1])

np.random.shuffle(face_data)
knn = KNN(task="R", data=face_data[:int(len(face_data) * .80)], k=5)
# knn.optimize_k(data_set=face_data)
knn.test(test_data=face_data[int(len(face_data) * .80):])

inp = face_data[-1][0]
predicted = knn.predict([inp])[0].reshape((112,46))
plt.imshow(np.concatenate([inp.reshape(112,46), predicted], axis=1), cmap="gray")
plt.show()
