from abc import ABC, abstractmethod
import numpy as np

class Regressor(ABC):
    @abstractmethod
    def fit(self, x_sample: np.ndarray, y_sample: np.ndarray):
        """サンプルに合わせて内部のパラメータを学習する

        Args:
            x_sample (np.ndarray): サンプルのx（1次元配列）
            y_sample (np.ndarray): サンプルのy（1次元配列）
        """
        ...

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """与えられたxに対する関数値を予測する

        Args:
            x (np.ndarray): 計算すべきxの値（1次元配列）

        Returns:
            np.ndarray: 予測値。入力xに対応した1次元配列
        """
        ...

class PolyRegressor(Regressor):
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

class GPRegressor(Regressor):
    def __init__(self, sigma_x, sigma_y):
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def fit(self, x_sample: np.ndarray, y_sample: np.ndarray):
        x_s = x_sample[:, np.newaxis]
        y_s = y_sample[:, np.newaxis]
        G = self._gaussian(x_s, x_s.T)
        sigma_I = self.sigma_y * np.eye(G.shape[0])
        self.x_s = x_s
        self.a = np.linalg.inv(G + sigma_I) @ y_s

    def predict(self, x):
        g = self._gaussian(x[:, np.newaxis], self.x_s.T)
        y_pred = np.squeeze(g @ self.a)
        return y_pred

    def _gaussian(self, col, row) -> np.ndarray:
        return np.exp(- (col - row) ** 2 / (2 * self.sigma_x ** 2))

def build_regressor(name, kwargs_all) -> Regressor:
    REGRESSORS = dict(
        poly=PolyRegressor,
        gp=GPRegressor,
    )
    regressor_cls = REGRESSORS[name]
    kwargs = kwargs_all[name]
    return regressor_cls(**kwargs)
