import time
from matplotlib.figure import Figure
import japanize_matplotlib as _
import numpy as np

def main():
    # 初期設定
    x = np.linspace(-1, 1, 101)
    y = np.sin(np.pi * x)
    y_range = np.max(y) - np.min(y)
    Ns = 2 ** np.arange(4, 25)
    n_repeat = 20
    ts = np.empty((Ns.size, n_repeat))
    ss = np.empty((Ns.size, n_repeat))
    d = 7
    p = np.arange(d+1)[np.newaxis, :]
    eps = 1e-8
    n_split = 16
    def quantitize(xs, ys, alpha=-1, beta=1):
        x_split = np.linspace(alpha, beta, n_split+1)[:, np.newaxis]
        x_filter = np.where(x_split > xs.T, 1, 0)
        y_map = x_filter[1:, :] - x_filter[:-1, :]
        y_map_sum = np.sum(y_map, axis=1, keepdims=True) 
        if np.all(y_map_sum):
            y_map = y_map / y_map_sum
            xs = (x_split[:-1] + x_split[1:]) / 2
            ys = y_map @ ys
        return xs, ys
    def non_quantitize(xs, ys):
        return xs, ys

    for i_N, N in enumerate(Ns):
        sample_x = np.random.uniform(-1, 1, (N, n_repeat))
        sample_noise = np.random.normal(0, y_range*0.05, (N, n_repeat) )
        sample_y = np.sin(np.pi * sample_x) + sample_noise
        for i_repeat in range(n_repeat):
            # 多項式フィッティング
            ## 学習サンプルから係数を求める
            t_start = time.time() # 測定開始
            _sample_x, _sample_y = quantitize(
                sample_x[:, [i_repeat]], sample_y[:, [i_repeat]]
            )
            sample_X = _sample_x ** p
            sample_XX_inv = np.linalg.inv(sample_X.T @ sample_X)
            a =  sample_XX_inv @ sample_X.T @ _sample_y
            t_end = time.time() # 測定終了
            ## 求めた係数を用いて y の値を予測
            X = x[:, np.newaxis] ** p
            y_pred = np.squeeze(X @ a)
            ts[i_N, i_repeat] = t_end - t_start
            # 評価指標の算出
            norm_y = np.sum(np.abs(y))
            norm_diff = np.sum(np.abs(y - y_pred))
            score = norm_diff / (norm_y + eps)
            ss[i_N, i_repeat] = score
        t = np.mean(ts[i_N, :])
        print(f'{N = }: {t = }')
        print(f'{score = :.3f}')
    ts = np.mean(ts, axis=1)
    ss = np.mean(ss, axis=1)
    # グラフの表示
    fig_t = Figure()
    ax = fig_t.add_subplot(
        1, 1, 1, xlabel='学習サンプル数 $N$', ylabel='パラメータ推定時間 $t$ (秒)'
    )
    ax.set_title(r'学習サンプル数$N$とパラメータ推定時間$t$の関係')
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.axhline(color="#777777")
    ax.plot(Ns, ts, label='真の関数 $f$')
    fig_t.savefig('time.png')
    fig_s = Figure()
    ax = fig_s.add_subplot(
        1, 1, 1, xlabel='学習サンプル数 $N$', ylabel='評価指標 $s$'
    )
    ax.set_title(r'学習サンプル数$N$と評価指標$s$の関係')
    ax.set_xscale('log')
    ax.set_yscale('linear')
    ax.axhline(color="#777777")
    ax.plot(Ns, ss, label='真の関数 $f$')
    fig_s.savefig('score.png')

if __name__ == '__main__':
    main()