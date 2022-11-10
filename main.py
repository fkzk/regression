import numpy as np ##
from matplotlib.figure import Figure

def main():
    # x, f(x)の準備 ##
    x = np.linspace(-1, 1, 101) ##
    y = np.sin(np.pi * x) ##
    # グラフの作成
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1) ##
    ax.plot(x, y) ##
    fig.savefig('out.png')

if __name__ == '__main__':
    main()