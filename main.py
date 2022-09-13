from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def sin(x: jnp.ndarray) -> jnp.ndarray:
    """sin(πx)を計算

    Args:
        x (jnp.ndarray): 計算すべきxが1次元に並んだ配列
    
    Returns:
        jnp.ndarray: sin(πx)の計算結果
    """
    return jnp.sin(jnp.pi*x)

def poly2(x: jnp.ndarray) -> jnp.ndarray:
    """x^2 + 0.5x - 0.7を計算

    Args:
        x (jnp.ndarray): 計算すべきxが1次元に並んだ配列

    Returns:
        jnp.ndarray: x^2 + 0.5x - 0.7 の計算結果
    """
    return x**2 + 0.5 * x - 0.7

TARGETS = dict(
    sin=sin,
    poly2=poly2,
)

def get_target_fn(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """nameに応じたターゲット関数t(x)を返す

    Args:
        name (str): ターゲット関数の名前

    Returns:
        Callable[[jnp.ndarray], jnp.ndarray]: nameに応じたターゲット関数
    """
    return TARGETS[name]

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
# サンプル点に対するx^pの計算
p = jnp.arange(n_degree+1)
X = jnp.power(sample_x[:, jnp.newaxis], p[jnp.newaxis, :])
# 係数の決定
I = jnp.eye(n_degree+1)
a = jnp.linalg.inv(c_l2 * I + X.T @ X) @ X.T @ sample_target[:, jnp.newaxis]
# すべてのxに対する多項式の計算
X_all = jnp.power(x[:, jnp.newaxis], p[jnp.newaxis, :])
poly_x = (X_all @ a)[:]

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