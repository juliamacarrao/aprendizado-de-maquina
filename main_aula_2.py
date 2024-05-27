from sklearn import datasets
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

x = iris.data
y = iris.target

X_train , X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=7, stratify=y)

knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))