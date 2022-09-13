from typing import Callable

import jax.numpy as jnp

def sin(x: jnp.ndarray) -> jnp.ndarray:
    """sin(πx)を計算

    Args:
        x (jnp.ndarray): 計算すべきxが1次元に並んだ配列
    
    Returns:
        jnp.ndarray: sin(πx)の計算結果
    """
    return jnp.sin(jnp.pi*x)

def poly2(x: jnp.ndarray) -> jnp.ndarray:
    """x^2 + 0.5x - 0.7を計算

    Args:
        x (jnp.ndarray): 計算すべきxが1次元に並んだ配列

    Returns:
        jnp.ndarray: x^2 + 0.5x - 0.7 の計算結果
    """
    return x**2 + 0.5 * x - 0.7

__TARGETS = dict(
    sin=sin,
    poly2=poly2,
)

def get_target_fn(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """nameに応じたターゲット関数t(x)を返す

    Args:
        name (str): ターゲット関数の名前

    Returns:
        Callable[[jnp.ndarray], jnp.ndarray]: nameに応じたターゲット関数
    """
    return __TARGETS[name]
