import numpy as np

class PolyRegressor:
    def __init__(self, d):
        self.d = d
        self.p = np.arange(d+1)[np.newaxis, :]

    def fit(self, x_sample, y_sample):
        ## Xを作る
        x_s = x_sample[:, np.newaxis]
        X_s = x_s ** self.p
        ## 係数aを求める
        y_s = y_sample[:, np.newaxis]
        X_inv = np.linalg.inv(X_s.T @ X_s)
        self.a = X_inv @ X_s.T @ y_s

    def predict(self, x):
        ## yの予測値を計算
        y_pred = np.squeeze((x[:, np.newaxis] ** self.p) @ self.a)
        return y_pred

class GPRegressor:
    def __init__(self, sigma_x, sigma_y):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def fit(self, x_sample: np.ndarray, y_sample: np.ndarray):
        x_s = x_sample[:, np.newaxis]
        y_s = y_sample[:, np.newaxis]
        self.x_s = x_s
        G = self._gaussian(x_s, x_s.T)
        sigma_I = self.sigma_y * np.eye(G.shape[0])
        self.a = np.linalg.inv(G + sigma_I) @ y_s

    def predict(self, x):
        g = self._gaussian(x[:, np.newaxis], self.x_s.T)
        y_pred = np.squeeze(g @ self.a)
        return y_pred

    def _gaussian(self, col: np.ndarray, row: np.ndarray) -> np.ndarray:
        return np.exp(- (col - row) ** 2 / (2 * (self.sigma_x ** 2)))

def build_regressor(regressor_name, regressor_kwargs):
    REGRESSORS = dict(
        poly = PolyRegressor,
        gp = GPRegressor,
    )
    regressor_cls = REGRESSORS[regressor_name]
    init_kwargs = regressor_kwargs[regressor_name]
    return regressor_cls(**init_kwargs)
