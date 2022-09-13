import jax.numpy as jnp
import matplotlib.pyplot as plt

fig = plt.Figure()
ax = fig.add_subplot(1, 1, 1)
x = jnp.linspace(-1, 1, num=101)
sinx = jnp.sin(jnp.pi*x)
ax.plot(x, sinx)
fig.savefig('sinx.png')