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
    ts = np.empty((4, Ns.size, n_repeat))
    ss = np.empty((4, Ns.size, n_repeat))
    d = 7
    p = np.arange(d+1)[np.newaxis, :]
    eps = 1e-8
    n_split = 32
    n_max = 65536
    def quantitize(xs, ys, alpha=-1, beta=1):
        x_split = np.linspace(alpha, beta, n_split+1)[:, np.newaxis]
        x_min = (x_split[0] + x_split[1]) / 2
        x_max = (x_split[-2] + x_split[-1]) / 2
        _xs = (np.squeeze(xs) - x_min) * ((n_split-1) / (x_max - x_min))
        _xs = np.round(_xs).astype(int)
        _ys = np.empty((n_split, 1))
        for i in range(n_split):
            __ys: np.ndarray = ys[_xs == i, 0]
            if __ys.size == 0:
                return xs, ys
            _ys[i, 0] = np.mean(__ys)
        xs = (x_split[:-1] + x_split[1:]) / 2
        return xs, _ys
    def non_quantitize(xs, ys):
        return xs, ys

    for i_split, use_split in enumerate((True, False)):
        preprocess = quantitize if use_split else non_quantitize
        for i_limit, use_limit_n_sample in enumerate((True, False)):
            i_cond = i_split * 2 + i_limit
            print(f'{use_limit_n_sample = }, {use_split = }')
            for i_N, N in enumerate(Ns):
                sample_x = np.random.uniform(-1, 1, (N, n_repeat))
                sample_noise = np.random.normal(0, y_range*0.05, (N, n_repeat) )
                sample_y = np.sin(np.pi * sample_x) + sample_noise
                if use_limit_n_sample and N > n_max:
                    sample_x = sample_x[:n_max, :]
                    sample_y = sample_y[:n_max, :]
                for i_repeat in range(n_repeat):
                    # 多項式フィッティング
                    ## 学習サンプルから係数を求める
                    t_start = time.time() # 測定開始
                    _sample_x, _sample_y = preprocess(
                        sample_x[:, [i_repeat]], sample_y[:, [i_repeat]]
                    )
                    sample_X = _sample_x ** p
                    sample_XX_inv = np.linalg.inv(sample_X.T @ sample_X)
                    a =  sample_XX_inv @ sample_X.T @ _sample_y
                    t_end = time.time() # 測定終了
                    ## 求めた係数を用いて y の値を予測
                    X = x[:, np.newaxis] ** p
                    y_pred = np.squeeze(X @ a)
                    ts[i_cond, i_N, i_repeat] = t_end - t_start
                    # 評価指標の算出
                    norm_y = np.sum(np.abs(y))
                    norm_diff = np.sum(np.abs(y - y_pred))
                    score = norm_diff / (norm_y + eps)
                    ss[i_cond, i_N, i_repeat] = score
                t = np.mean(ts[i_cond, i_N, :])
                print(f'{N = }: {t = }')
                print(f'{score = :.3f}')
    ts = np.mean(ts, axis=2)
    ss = np.mean(ss, axis=2)
    # グラフの表示
    fig_t = Figure()
    ax = fig_t.add_subplot(
        1, 1, 1, xlabel='学習サンプル数 $N$', ylabel='パラメータ推定時間 $t$ (秒)'
    )
    ax.set_title(r'学習サンプル数$N$とパラメータ推定時間$t$の関係')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axhline(color="#777777")
    for i_split, use_split in enumerate((False, True)):
        for i_limit, use_limit_n_sample in enumerate((False, True)):
            i_cond = (1-i_split) * 2 + (1-i_limit)
            label_limit = f'$N_{{\\mathrm{{max}}}}={n_max}$'
            label_split = f'$N_{{\\mathrm{{split}}}}={n_split}$'
            if use_split and use_limit_n_sample:
                label = f'{label_limit}, {label_split}'
            elif not use_split and use_limit_n_sample:
                label = f'{label_limit}'
            elif use_split and not use_limit_n_sample:
                label = f'{label_split}'
            else:
                label = f'ベースライン'
            ax.plot(Ns, ts[i_cond], label=label)
            if not use_split and not use_limit_n_sample:
                fig_t.savefig('time_baseline.png')
    ax.legend()
    fig_t.savefig('time.png')
    fig_s = Figure()
    ax = fig_s.add_subplot(
        1, 1, 1, xlabel='学習サンプル数 $N$', ylabel='評価指標 $s$'
    )
    ax.set_title(r'学習サンプル数$N$と評価指標$s$の関係')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.axhline(color="#777777")
    for i_split, use_split in enumerate((False, True)):
        for i_limit, use_limit_n_sample in enumerate((False, True)):
            i_cond = (1-i_split) * 2 + (1-i_limit)
            label_limit = f'$N_{{\\mathrm{{max}}}}={n_max}$'
            label_split = f'$N_{{\\mathrm{{split}}}}={n_split}$'
            if use_split and use_limit_n_sample:
                label = f'{label_limit}, {label_split}'
            elif not use_split and use_limit_n_sample:
                label = f'{label_limit}'
            elif use_split and not use_limit_n_sample:
                label = f'{label_split}'
            else:
                label = f'ベースライン'
            ax.plot(Ns, ss[i_cond], label=label)
            if not use_split and not use_limit_n_sample:
                fig_s.savefig('score_baseline.png')
    ax.legend()
    fig_s.savefig('score.png')
    #fig = Figure()
    #ax = fig.add_subplot(1, 1, 1, xlabel='$x$', ylabel='$y$')
    #ax.set_title(r'$y = \sin (\pi x)$')
    #ax.axhline(0, color='#777777')
    #ax.axvline(0, color='#777777')
    #for x__ in np.linspace(-1, 1, n_split+1):
    #    ax.axvline(x__, color='#77aa77')
    #ax.plot(x, y, label='真の関数 $f$')
    #ax.scatter(sample_x, sample_y, color='#ffdddd', label='学習サンプル')
    #ax.scatter(_sample_x, _sample_y, color='red', label='学習サンプルの区間平均')
    #ax.plot(x, y_pred, label=r'回帰関数 $\hat{f}$')
    #ax.legend()
    #fig.savefig('out.png')

if __name__ == '__main__':
    main()