import numpy as np
from matplotlib.figure import Figure
import japanize_matplotlib as _

from regressor import build_regressor

def calculate_score(y, y_pred, eps_score):
    norm_diff = np.sum(np.abs(y - y_pred))
    norm_y = np.sum(np.abs(y))
    score = norm_diff / (norm_y + eps_score)
    return score

def save_graph(
    xy = None,
    xy_sample = None,
    xy_pred = None,
    title = None,
    filename = 'out.png',
):
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.axhline(color='#777777')
    ax.axvline(color='#777777')
    if xy is not None:
        x, y = xy # (x, y)のアンパック
        ax.plot(x, y, color='C0', label='真の関数 $f$')
    if xy_sample is not None:
        x_sample, y_sample = xy_sample
        ax.scatter(x_sample, y_sample, color='red', label='学習サンプル')
    if xy_pred is not None:
        x, y_pred = xy_pred
        ax.plot(x, y_pred, color='C1', label='回帰関数 $\\hat{f}$')
    ax.legend()
    fig.savefig(filename)

def main():
    # 実験条件
    x_min = -1
    x_max = 1
    n_train = 20
    n_test = 101
    noise_ratio = 0.05
    eps_score = 1e-8
    # 回帰分析に関する設定
    regressor_name = 'poly'
    regressor_kwargs = dict(
        poly = dict(
            d = 3,
        ),
    )
    regressor = build_regressor(regressor_name, regressor_kwargs)
    # x, f(x)の準備
    x = np.linspace(x_min, x_max, n_test)
    y = np.sin(np.pi * x)
    # サンプルの準備
    x_sample = np.random.uniform(x_min, x_max, (n_train, ))
    range_y = np.max(y) - np.min(y)
    noise_sample = np.random.normal(0, range_y*noise_ratio, (n_train, ))
    y_sample = np.sin(np.pi * x_sample) + noise_sample
    # 多項式フィッティング
    regressor.fit(x_sample, y_sample)
    y_pred = regressor.predict(x)
    # 評価指標の計算
    score = calculate_score(y, y_pred, eps_score)
    print(f'{score = :.3f}')
    # グラフの作成
    save_graph(
        xy=(x, y), xy_sample=(x_sample, y_sample), xy_pred=(x, y_pred),
        title=r'$y = \sin (\pi x)$'
    )

if __name__ == '__main__':
    main()