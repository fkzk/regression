import numpy as np

class PolyRegressor:
    def __init__(self, d):
        self.d = d
        self.p = np.arange(d+1)[np.newaxis, :] # 1 x (d+1)

    def fit(self, x_sample, y_sample):
        X_sample = x_sample[:, np.newaxis] ** self.p
        XX_inv_sample = np.linalg.inv(X_sample.T @ X_sample)
        self.a =  XX_inv_sample @ X_sample.T @ y_sample[:, np.newaxis]

    def predict(self, x):
        X = x[:, np.newaxis] ** self.p
        y_pred = np.squeeze(X @ self.a)
        return y_pred

