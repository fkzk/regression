import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from target_fn import get_target_fn

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

# 値の初期化
x_max = 1
n_all = 101
n_train = 10
n_degree = 11
random_seed = 0
c_l2 = 0.005
noise_rate = 0.05
target_name = 'sin'

# ターゲット関数 t の取得
target_fn = get_target_fn(target_name)
# y = t(x)の計算
x = jnp.linspace(-x_max, x_max, num=n_all)
target = target_fn(x)
# サンプル点の選出
random_key = jax.random.PRNGKey(random_seed)
sample_x = jax.random.uniform(
    random_key, shape=(n_train,), minval=-x_max, maxval=x_max
)
# サンプル点に対する t(x_i)の計算
sample_target = target_fn(sample_x)
# 学習用サンプルにノイズを加える
sigma = (jnp.max(target) - jnp.min(target)) * noise_rate
_, noise_key = jax.random.split(random_key)
noise = jax.random.normal(noise_key, shape=sample_x.shape) * sigma
sample_target = sample_target + noise
# 回帰の計算
regressor = PolyRegressor(n_degree, c_l2)
regressor.train(sample_x, sample_target)
poly_x = regressor(x)

# 評価値の計算・表示
mae = jnp.mean(jnp.abs(target-poly_x))
nmae = mae / (jnp.max(target) - jnp.min(target))
score = - 20 * jnp.log10(nmae)
print(f'近似スコア: {score:.2f} dB')

# 図とグラフの作成
fig = plt.Figure()
ax = fig.add_subplot(1, 1, 1, xlabel='x', ylabel='y')
# グラフの見た目の設定
ax.set_title(f'$y = ${target_name}$ (x)$')
ax.axhline(0, color='#777777')
ax.axvline(0, color='#777777')
ax.tick_params(direction='in', bottom=True, top=True, left=True, right=True)
# 表示範囲の設定
ax.set_xlim(-x_max, x_max)
y_min = jnp.min(target)
y_max = jnp.max(target)
ylim_min = y_min - (y_max-y_min) * 0.15
ylim_max = y_max + (y_max-y_min) * 0.15
ax.set_ylim(ylim_min, ylim_max)
# グラフのプロット
ax.plot(x, target, label='target')
ax.scatter(
    sample_x, sample_target, color='red', zorder=2, label='training sample'
)
ax.plot(x, poly_x, label=f'predicted ($d={n_degree}, \lambda={c_l2}$)')
# 凡例の表示・図の出力
ax.legend()
fig.savefig('result.png')