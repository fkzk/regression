import jax.numpy as jnp

class PolyRegressor:
    """多項式フィッティングで回帰するためのクラス
    """
    def __init__(self, n_degree: int, c_l2: float) -> None:
        """初期化

        Args:
            n_degree (int): 回帰に使う多項式の次数
            c_l2 (float): L2正則化の正則化係数
        """
        self.n_degree = n_degree
        self.p = jnp.arange(n_degree+1)
        self.c_l2 = c_l2
        self.a = None

    def train(self, sample_x: jnp.ndarray, sample_target: jnp.ndarray) -> None:
        """入出力のサンプルから多項式の係数を決定する

        Args:
            sample_x (jnp.ndarray): サンプルの入力となった1次元配列
            sample_target (jnp.ndarray): sample_xに対応する出力
        """
        # サンプル点に対するx^pの計算
        X = jnp.power(sample_x[:, jnp.newaxis], self.p[jnp.newaxis, :])
        # 係数の決定
        I = jnp.eye(self.n_degree+1)
        self.a = (
            jnp.linalg.inv(self.c_l2 * I + X.T @ X) @ X.T
            @ sample_target[:, jnp.newaxis]
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """多項式の計算をする

        Args:
            x (jnp.ndarray): 多項式に入力する1次元配列

        Returns:
            jnp.ndarray: 多項式の計算結果。xと同じ要素数の1次元配列
        """
        X = jnp.power(x[:, jnp.newaxis], self.p[jnp.newaxis, :])
        y_pred = (X @ self.a)[:]
        return y_pred

    def __str__(self) -> str:
        args = [
            f'$d={self.n_degree}$',
            f'$\lambda={self.c_l2}$',
        ]
        return f'Poly({", ".join(args)})'

class GPRegressor:
    def __init__(self, beta: float, sigma: float) -> None:
        self.beta = beta
        self.sigma = sigma
        self.sample_x = None

    def train(self, sample_x: jnp.ndarray, sample_target: jnp.ndarray) -> None:
        self.sample_x = sample_x
        K = self.kernel(sample_x, sample_x)
        I = jnp.eye(sample_x.shape[0])
        c = jnp.linalg.inv(K + I / self.beta) @ sample_target[:, jnp.newaxis]
        self.c = c

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        k = self.kernel(x, self.sample_x)
        mu = (k @ self.c)[:]
        return mu

    def kernel(self, x_row: jnp.ndarray, x_col: jnp.ndarray) -> jnp.ndarray:
        diff = x_row[:, jnp.newaxis] - x_col[jnp.newaxis, :]
        gaussian = jnp.exp(-(diff**2)/(2*self.sigma**2))
        return gaussian

    def __str__(self) -> str:
        args = [
            f'$\\beta={self.beta}$',
            f'$\sigma={self.sigma}$',
        ]
        return f'GP({", ".join(args)})'

__REGRESSORS = dict(
    poly=PolyRegressor,
    gp=GPRegressor,
)

def build_regressor(name, init_cfg):
    regressor_cls = __REGRESSORS[name]
    init_kwargs = init_cfg[name]
    regressor = regressor_cls(**init_kwargs)
    return regressor
