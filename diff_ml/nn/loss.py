import jax.numpy as jnp
from jaxtyping import Array, Float


# def mse(y: Float[Array, " n"], pred_y: Float[Array, " n"]) -> Float:
def mse(y, pred_y):
    """Mean squared error loss."""
    return jnp.mean((y - pred_y) ** 2)


def rmse(y: Float[Array, " n"], pred_y: Float[Array, " n"]) -> Float:
    """Root mean squared error loss."""
    return jnp.sqrt(mse(y, pred_y))
