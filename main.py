import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# 値の初期化
n = 3

# y = sin(πx)の計算
x = jnp.linspace(-1, 1, num=101)
sinx = jnp.sin(jnp.pi*x)
# サンプル点の選出
random_key = jax.random.PRNGKey(0)
sample_x = jax.random.uniform(random_key, shape=(10,), minval=-1, maxval=1)
# サンプル点に対する sin(πx_i)の計算
sample_sinx = jnp.sin(jnp.pi*sample_x)
# サンプル点に対するx^pの計算
p = jnp.arange(n+1)
X = jnp.power(sample_x[:, jnp.newaxis], p[jnp.newaxis, :])
# 係数の決定
a = jnp.linalg.inv(X.T @ X) @ X.T @ sample_sinx[:, jnp.newaxis]
# すべてのxに対する多項式の計算
X_all = jnp.power(x[:, jnp.newaxis], p[jnp.newaxis, :])
poly_x = (X_all @ a)[:]

# 図とグラフの作成
fig = plt.Figure()
ax = fig.add_subplot(1, 1, 1, xlabel='x', ylabel='y')
# グラフの見た目の設定
ax.set_title(r'$y = \sin (\pi x)$')
ax.axhline(0, color='#777777')
ax.axvline(0, color='#777777')
ax.tick_params(direction='in', bottom=True, top=True, left=True, right=True)
# グラフのプロット
ax.plot(x, sinx, label='target')
ax.scatter(sample_x, sample_sinx, color='red', zorder=2, label='training sample')
ax.plot(x, poly_x, label='predicted')
# 凡例の表示・図の出力
ax.legend()
fig.savefig('sinx.png')