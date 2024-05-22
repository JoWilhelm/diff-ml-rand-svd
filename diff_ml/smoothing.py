from collections.abc import Callable

import jax.numpy as jnp


def sigmoidal_smoothing(f1: Callable, f2: Callable, p=0.0, w=0.005):
    """Smoothly transition between two functions using sigmoidal smoothing.

    Args:
        f1 (callable): The first function, f1(x), where x is a real number.
        f2 (callable): The second function, f2(x), where x is a real number.
        p (float, optional): The position to transition between the two functions. Default is 0.0.
        w (float, optional): The width of the smoothing. Default is 0.005.

    Returns:
        callable: A function that smoothly transitions between f1 and f2 at point p.

    Example:
        zero_fn = lambda x: jnp.zeros_like(x)
        id_fn   = lambda x: x
        smooth_fn = sigmoidal_smoothing(zero_fn, id_fn, p=1.0, w=0.01)
        result = smooth_fn(0.5)
    """

    def sigma(x):
        """Sigmoid function for smoothing.

        Args:
            x (float): The input value.

        Returns:
            float: The result of the sigmoid function.
        """
        return 1.0 / (1.0 + jnp.exp(-(x - p) / w))

    def smooth_f(x):
        """Smoothed function.

        Args:
            x (float): The input value.

        Returns:
            float: The result of the smoothed function.
        """
        return (1 - sigma(x)) * f1(x) + sigma(x) * f2(x)

    return smooth_f
