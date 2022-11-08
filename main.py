from matplotlib.figure import Figure
import numpy as np

def main():
    x = np.linspace(-1, 1, 101)
    y = np.sin(np.pi * x)
    sample_x = np.random.uniform(-1, 1, (20, ))
    sample_y = np.sin(np.pi * sample_x)
    fig = Figure()
    ax = fig.add_subplot(1, 1, 1, xlabel='$x$', ylabel='$y$')
    ax.set_title(r'$y = \sin (\pi x)$')
    ax.axhline(0, color='#777777')
    ax.axvline(0, color='#777777')
    ax.plot(x, y)
    ax.scatter(sample_x, sample_y, color='red')
    fig.savefig('out.png')

if __name__ == '__main__':
    main()