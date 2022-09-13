import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# 値の初期化
x_max = 1
n_all = 101
n_train = 10
n_degree = 11
random_seed = 0
c_l2 = 0.005
noise_rate = 0.05

# y = sin(πx)の計算
x = jnp.linspace(-x_max, x_max, num=n_all)
sinx = jnp.sin(jnp.pi*x)
# サンプル点の選出
random_key = jax.random.PRNGKey(random_seed)
sample_x = jax.random.uniform(
    random_key, shape=(n_train,), minval=-x_max, maxval=x_max
)
# サンプル点に対する sin(πx_i)の計算
sample_sinx = jnp.sin(jnp.pi*sample_x)
# 学習用サンプルにノイズを加える
sigma = (jnp.max(sinx) - jnp.min(sinx)) * noise_rate
_, noise_key = jax.random.split(random_key)
noise = jax.random.normal(noise_key, shape=sample_x.shape) * sigma
sample_sinx = sample_sinx + noise
# サンプル点に対するx^pの計算
p = jnp.arange(n_degree+1)
X = jnp.power(sample_x[:, jnp.newaxis], p[jnp.newaxis, :])
# 係数の決定
I = jnp.eye(n_degree+1)
a = jnp.linalg.inv(c_l2 * I + X.T @ X) @ X.T @ sample_sinx[:, jnp.newaxis]
# すべてのxに対する多項式の計算
X_all = jnp.power(x[:, jnp.newaxis], p[jnp.newaxis, :])
poly_x = (X_all @ a)[:]

# 評価値の計算・表示
mae = jnp.mean(jnp.abs(sinx-poly_x))
nmae = mae / (jnp.max(sinx) - jnp.min(sinx))
score = - 20 * jnp.log10(nmae)
print(f'近似スコア: {score:.2f} dB')

# 図とグラフの作成
fig = plt.Figure()
ax = fig.add_subplot(1, 1, 1, xlabel='x', ylabel='y')
# グラフの見た目の設定
ax.set_title('$y = \sin (\pi x)$')
ax.axhline(0, color='#777777')
ax.axvline(0, color='#777777')
ax.tick_params(direction='in', bottom=True, top=True, left=True, right=True)
# 表示範囲の設定
ax.set_xlim(-x_max, x_max)
y_min = jnp.min(sinx)
y_max = jnp.max(sinx)
ylim_min = y_min - (y_max-y_min) * 0.15
ylim_max = y_max + (y_max-y_min) * 0.15
ax.set_ylim(ylim_min, ylim_max)
# グラフのプロット
ax.plot(x, sinx, label='target')
ax.scatter(sample_x, sample_sinx, color='red', zorder=2, label='training sample')
ax.plot(x, poly_x, label=f'predicted ($d={n_degree}, \lambda={c_l2}$)')
# 凡例の表示・図の出力
ax.legend()
fig.savefig('sinx.png')