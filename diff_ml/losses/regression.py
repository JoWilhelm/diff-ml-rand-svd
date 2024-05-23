from collections.abc import Callable
from enum import Enum

import equinox as eqx
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float


def mse(y: Float[Array, " n"], pred_y: Float[Array, " n"]) -> Float[Array, ""]:
    """Mean squared error loss."""
    return jnp.mean((y - pred_y) ** 2)


def rmse(y: Float[Array, " n"], pred_y: Float[Array, " n"]) -> Float[Array, ""]:
    """Root mean squared error loss."""
    return jnp.sqrt(mse(y, pred_y))


RegressionLossFn = Callable[..., Float[Array, ""]]


class SobolevLossType(Enum):
    FIRST_ORDER = 1
    SECOND_ORDER_HUTCHINSON = 2
    SECOND_ORDER_PCA = 3


# NOTE: This currently assumes that we have a regression problem
def sobolev(loss_fn: RegressionLossFn, *, method: SobolevLossType = SobolevLossType.FIRST_ORDER) -> RegressionLossFn:
    sobolev_loss_fn = loss_fn
    if method == SobolevLossType.FIRST_ORDER:

        def sobolev_first_order_loss(model, batch) -> Float[Array, ""]:
            x, y, dydx = batch["x"], batch["y"], batch["dydx"]
            y_pred, dydx_pred = vmap(eqx.filter_value_and_grad(model))(x)

            assert y.shape == y_pred.shape
            assert dydx.shape == dydx_pred.shape

            value_loss = loss_fn(y, y_pred)
            grad_loss = loss_fn(dydx, dydx_pred)

            alpha = 0.5
            beta = 0.5

            return alpha * value_loss + beta * grad_loss

        sobolev_loss_fn = sobolev_first_order_loss
    elif method == SobolevLossType.SECOND_ORDER_HUTCHINSON:
        raise NotImplementedError
    elif method == SobolevLossType.SECOND_ORDER_PCA:
        raise NotImplementedError

    return sobolev_loss_fn
