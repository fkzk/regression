from matplotlib.figure import Figure
import japanize_matplotlib as _
import numpy as np

def calculate_score(y, y_pred, score_eps):
    norm_y = np.sum(np.abs(y))
    norm_diff = np.sum(np.abs(y - y_pred))
    score = norm_diff / (norm_y + score_eps)
    return score

def save_graph(
    xy = None,
    xy_sample = None,
    xy_pred = None,
    title = None,
    filename = 'out.png',
):
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1, xlabel='$x$', ylabel='$y$')
    if title is not None:
        ax.set_title(title)
    ax.axhline(0, color='#777777')
    ax.axvline(0, color='#777777')
    if xy is not None:
        x, y = xy
        ax.plot(x, y, label='真の関数 $f$')
    if xy_sample is not None:
        x_sample, y_sample = xy_sample
        ax.scatter(x_sample, y_sample, color='red', label='学習サンプル')
    if xy_pred is not None:
        x_pred, y_pred = xy_pred
        ax.plot(x_pred, y_pred, label=r'回帰関数 $\hat{f}$')
    ax.legend()
    fig.savefig(filename)

def main():
    # 実験条件
    x_min = -1
    x_max = 1
    n_train = 20
    n_test = 101
    noise_ratio = 0.05
    score_eps = 1e-8
    # 多項式フィッティングの設定
    d = 3
    # 変数の準備
    x = np.linspace(x_min, x_max, n_test)
    y = np.sin(np.pi * x)
    y_range = np.max(y) - np.min(y)
    sample_x = np.random.uniform(x_min, x_max, (n_train, ))
    sample_noise = np.random.normal(0, y_range*noise_ratio, (n_train, ) )
    sample_y = np.sin(np.pi * sample_x) + sample_noise
    # 多項式フィッティング
    ## 学習サンプルから係数を求める
    p = np.arange(d+1)
    sample_X = sample_x[:, np.newaxis] ** p[np.newaxis, :]
    sample_XX_inv = np.linalg.inv(sample_X.T @ sample_X)
    a =  sample_XX_inv @ sample_X.T @ sample_y[:, np.newaxis]
    ## 求めた係数を用いて y の値を予測
    X = x[:, np.newaxis] ** p[np.newaxis, :]
    y_pred = np.squeeze(X @ a)
    # 評価指標の算出
    score = calculate_score(y, y_pred, score_eps)
    print(f'{score = :.3f}')
    # グラフの表示
    save_graph(
        xy=(x, y), xy_sample=(sample_x, sample_y), xy_pred=(x, y_pred),
        title=r'$y = \sin (\pi x)$'
    )

if __name__ == '__main__':
    main()