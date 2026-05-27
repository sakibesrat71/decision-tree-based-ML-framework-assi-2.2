import numpy as np

class LinearRegression:
    def _init_(self):
        self.weights = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # add bias term (column of 1s)
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]

        self.weights = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y

    def predict(self, X):
        X = np.array(X)
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]

        return X_bias @ self.weights