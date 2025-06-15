import numpy as np


# Linear Regression for a Linear Function
def linear_regression(X, y):
    x_b = np.c_[np.ones((X.shape[0], 1)), X.reshape(X.shape[0], -1)]
    theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
    return theta_best


# Predicting new values using the learned parameters
def predict(X, theta):
    x_b = np.c_[np.ones((X.shape[0], 1)), X.reshape(X.shape[0], -1)]
    return x_b.dot(theta)


if __name__ == "__main__":
    X = np.array([[1], [2], [3]])
    y = np.array([7, 8, 9])

    theta = linear_regression(X, y)
    print("Theta:", theta)

    X_new = np.array([[4], [5], [6]])
    predictions = predict(X_new, theta)
    print("Predictions:", predictions)
