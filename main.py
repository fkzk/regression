import jax
import jax.numpy as jnp
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from regressor import build_regressor
from target_fn import get_target_fn

# 値の初期化
x_max = 1
n_all = 101
n_train = 10
random_seed = 0
noise_rate = 0.05
target_name = 'sin'
regressor_name = ['poly', 'gp']
regressor_cfg = dict(
    poly = dict(
        n_degree = 11,
        c_l2 = 0.005,
    ),
    gp = dict(
        beta = 10,
        sigma = 0.25,
    )
)

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
if isinstance(regressor_name, str):
    regressor_name = [regressor_name, ]
regressor = dict()
y_pred = dict()
for reg_name in regressor_name:
    reg = build_regressor(reg_name, regressor_cfg)
    regressor[reg_name] = reg
    reg.train(sample_x, sample_target)
    y_pred[reg_name] = reg(x)

# 評価値の計算・表示
for reg_name in regressor_name:
    mae = jnp.mean(jnp.abs(target-y_pred[reg_name]))
    nmae = mae / (jnp.max(target) - jnp.min(target))
    score = - 20 * jnp.log10(nmae)
    print(f'** {regressor[reg_name]} **')
    print(f'近似スコア: {score:.2f} dB')

# 図とグラフの作成
fig: Figure = plt.Figure()
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
for reg_name in regressor_name:
    ax.plot(x, y_pred[reg_name], label=f'predicted {regressor[reg_name]}')
# 凡例の表示・図の出力
ax.legend()
fig.savefig('result.png')