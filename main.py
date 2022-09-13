import jax.numpy as jnp
import matplotlib.pyplot as plt

fig = plt.Figure()
ax = fig.add_subplot(1, 1, 1, xlabel='x', ylabel='y')
ax.set_title(r'$y = \sin (\pi x)$')
ax.axhline(0, color='#777777')
ax.axvline(0, color='#777777')
ax.tick_params(direction='in', bottom=True, top=True, left=True, right=True)
x = jnp.linspace(-1, 1, num=101)
sinx = jnp.sin(jnp.pi*x)
ax.plot(x, sinx)
fig.savefig('sinx.png')