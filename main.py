import numpy as np
from matplotlib.figure import Figure
import japanize_matplotlib as _

def main():
    # 実験条件
    x_min = -1
    x_max = 1
    n_train = 20
    n_test = 101
    noise_ratio = 0.05
    eps_score = 1e-8
    # 多項式フィッティングの設定
    d = 3
    # x, f(x)の準備
    x = np.linspace(x_min, x_max, n_test)
    y = np.sin(np.pi * x)
    # サンプルの準備
    x_sample = np.random.uniform(x_min, x_max, (n_train, ))
    range_y = np.max(y) - np.min(y)
    noise_sample = np.random.normal(0, range_y*noise_ratio, (n_train, ))
    y_sample = np.sin(np.pi * x_sample) + noise_sample
    # 多項式フィッティング
    ## Xを作る
    p = np.arange(d+1)[np.newaxis, :]
    x_s = x_sample[:, np.newaxis]
    X_s = x_s ** p
    ## 係数aを求める
    y_s = y_sample[:, np.newaxis]
    X_inv = np.linalg.inv(X_s.T @ X_s)
    a = X_inv @ X_s.T @ y_s
    ## yの予測値を計算
    y_pred = np.squeeze((x[:, np.newaxis] ** p) @ a)
    # 評価指標の計算
    norm_diff = np.sum(np.abs(y - y_pred))
    norm_y = np.sum(np.abs(y))
    score = norm_diff / (norm_y + eps_score)
    print(f'{score = :.3f}')
    # グラフの作成
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('$y = \\sin (\\pi x)$')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.axhline(color='#777777')
    ax.axvline(color='#777777')
    ax.plot(x, y, label='真の関数 $f$')
    ax.scatter(x_sample, y_sample, color='red', label='学習サンプル')
    ax.plot(x, y_pred, label='回帰関数 $\\hat{f}$')
    ax.legend()
    fig.savefig('out.png')

if __name__ == '__main__':
    main()