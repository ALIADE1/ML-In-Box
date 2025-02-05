import numpy as np
from collections import Counter  # num of occurrences of elements in an iterable
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


class KNN:

    def __init__(self, K=3, task="classification"):
        self.K = K
        self.task = task

    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self._predict(x))
        return np.array(predictions)

    def _predict(self, x):
        distances = np.sum((self.X_train - x) ** 2, axis=1)
        indices = np.argsort(distances)[: self.K]
        kn_outputs = [self.Y_train[i] for i in indices]

        if self.task == "classification":
            most_common = Counter(kn_outputs).most_common(1)
            return most_common[0][0]
        elif self.task == "regression":
            return np.mean(kn_outputs)


# Classification Task
X_Classification, Y_Classification = make_classification(
    n_samples=100, n_features=7, n_clusters_per_class=2, n_classes=2, random_state=42
)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_Classification, Y_Classification, test_size=0.2, random_state=42
)

knn_classifier = KNN(K=5, task="classification")
knn_classifier.fit(X_train, Y_train)
Y_pred = knn_classifier.predict(X_test)

Acc = accuracy_score(Y_test, Y_pred)
print(f"Classification Accuracy: {Acc * 100:0.2f}%")


# Regression Task
X_regression, Y_regression = make_regression(
    n_samples=500, n_features=2, n_informative=10, noise=0.1, random_state=42
)

X_train, X_test, Y_train, Y_test = train_test_split(
    X_regression, Y_regression, test_size=0.2, random_state=42
)

knn_regressor = KNN(K=5, task="regression")
knn_regressor.fit(X_train, Y_train)
Y_pred = knn_regressor.predict(X_test)

MSE = mean_squared_error(Y_test, Y_pred)
print(f"Regression MSE: {MSE:0.2f}")
