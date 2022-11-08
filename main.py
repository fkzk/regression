from matplotlib.figure import Figure
import japanize_matplotlib as _
import numpy as np

def main():
    x = np.linspace(-1, 1, 101)
    y = np.sin(np.pi * x)
    y_range = np.max(y) - np.min(y)
    sample_x = np.random.uniform(-1, 1, (20, ))
    sample_noise = np.random.normal(0, y_range*0.05, (20, ) )
    sample_y = np.sin(np.pi * sample_x) + sample_noise
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1, xlabel='$x$', ylabel='$y$')
    ax.set_title(r'$y = \sin (\pi x)$')
    ax.axhline(0, color='#777777')
    ax.axvline(0, color='#777777')
    ax.plot(x, y, label='真の関数 $f$')
    ax.scatter(sample_x, sample_y, color='red', label='学習サンプル')
    ax.legend()
    fig.savefig('out.png')

if __name__ == '__main__':
    main()