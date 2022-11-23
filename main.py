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

class PolyRegressor:
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
    regressor = PolyRegressor(d)
    # 変数の準備
    x = np.linspace(x_min, x_max, n_test)
    y = np.sin(np.pi * x)
    range_y = np.max(y) - np.min(y)
    x_sample = np.random.uniform(x_min, x_max, (n_train, ))
    noise_sample = np.random.normal(0, range_y*noise_ratio, (n_train, ) )
    y_sample = np.sin(np.pi * x_sample) + noise_sample
    # 多項式フィッティング
    ## 学習サンプルから係数を求める
    regressor.fit(x_sample, y_sample)
    ## 求めた係数を用いて y の値を予測
    y_pred = regressor.predict(x)
    # 評価指標の算出
    score = calculate_score(y, y_pred, score_eps)
    print(f'{score = :.3f}')
    # グラフの表示
    save_graph(
        xy=(x, y), xy_sample=(x_sample, y_sample), xy_pred=(x, y_pred),
        title=r'$y = \sin (\pi x)$'
    )

if __name__ == '__main__':
    main()