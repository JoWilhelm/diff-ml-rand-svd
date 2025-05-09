from collections.abc import Callable
from enum import Enum

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float

from diff_ml.hvp_stuff import get_HVPs


RegressionLossFn = Callable[..., Float[Array, ""]]


@jax.named_scope("dml.losses.mse")
def mse(y: Float[Array, " n"], pred_y: Float[Array, " n"]) -> Float[Array, ""]:
    """Mean squared error loss."""
    return jnp.mean((y - pred_y) ** 2)


@jax.named_scope("dml.losses.rmse")
def rmse(y: Float[Array, " n"], pred_y: Float[Array, " n"]) -> Float[Array, ""]:
    """Root mean squared error loss."""
    return jnp.sqrt(mse(y, pred_y))


# TODO understand do we need this?
class MakeScalar(eqx.Module):
    model: eqx.Module
    def __call__(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        return jnp.reshape(out, ())


class SobolevLossType(Enum):
    """Types of Sobolev loss to use.

    Attributes:
        ZEROTH_ORDER: Unmodified loss function.
        FIRST_ORDER: Use first-order derivative information.
        SECOND_ORDER_HUTCHINSON: Use second-order hessian-vector products sampled in random directions.
        SECOND_ORDER_PCA: Use second-order hessian-vector products sampled in PCA directions.

    """

    ZEROTH_ORDER = 0
    FIRST_ORDER = 1
    SECOND_ORDER_HUTCHINSON = 2
    SECOND_ORDER_PCA = 3


@jax.named_scope("dml.losses.sobolev")
def sobolev(loss_fn: RegressionLossFn, *, method: SobolevLossType = SobolevLossType.FIRST_ORDER) -> RegressionLossFn:
    sobolev_loss_fn = loss_fn

    if method == SobolevLossType.FIRST_ORDER:

        def loss_balance(n_dims: int, weighting: float = 1.0) -> tuple[float, float]:
            lambda_scale = weighting * n_dims
            n_elements = 1.0 + lambda_scale
            alpha = 1.0 / n_elements
            beta = lambda_scale / n_elements
            return alpha, beta

        def sobolev_first_order_loss(model, batch) -> Float[Array, ""]:
            x, y, dydx = batch["x"], batch["y"], batch["dydx"]
            y_pred, dydx_pred = vmap(eqx.filter_value_and_grad(model))(x)

            assert y.shape == y_pred.shape
            assert dydx.shape == dydx_pred.shape

            value_loss = loss_fn(y, y_pred)
            grad_loss = loss_fn(dydx, dydx_pred)

            n_dims = x.shape[-1]
            alpha, beta = loss_balance(n_dims)

            return alpha * value_loss + beta * grad_loss

        sobolev_loss_fn = sobolev_first_order_loss
    elif method == SobolevLossType.SECOND_ORDER_HUTCHINSON:
        raise NotImplementedError
    


    elif method == SobolevLossType.SECOND_ORDER_PCA:
        


        #def loss_balance(n_dims: int, weighting: float = 1.0) -> tuple[float, float]:
        #    lambda_scale = weighting * n_dims
        #    n_elements = 1.0 + lambda_scale
        #    alpha = 1.0 / n_elements
        #    beta = lambda_scale / n_elements
        #    return alpha, beta



        def sobolev_second_order_loss(model, batch) -> Float[Array, ""]:

            # unpack dictionary for readability
            x, y, dydx = batch["x"], batch["y"], batch["dydx"]
            
            # TODO alpha, beta, gamma = loss_balance()
            
            # get surrogate prediction, first-order derivative and hessian
            y_pred, dydx_pred = vmap(eqx.filter_value_and_grad(MakeScalar(model)))(x)
            hessian = eqx.filter_vmap(jax.hessian(MakeScalar(model)))(x) # TODO move lower where needed?
            assert y.shape == y_pred.shape
            assert dydx.shape == dydx_pred.shape
            
            # TODO understand batch hmp? needed for what?

            # apply PCA to first-order gradients of predictions
            dydx_used = dydx_pred
            #dydx_used = dydx               # alternatively use dydx of reference model
            #dydx_used = dydx_pred - dydx   # ore difference between the two
            dydx_means = jnp.mean(dydx_used, axis=0)
            tiled_dydx_used_means = jnp.tile(dydx_means, (dydx_used.shape[0], 1))
            dydx_used_mean_adjusted = dydx_used - tiled_dydx_used_means
            U, S, VT = jnp.linalg.svd(dydx_used_mean_adjusted, full_matrices=False)
            principal_components = jnp.diag(S) @ VT

            # select PCs that account for 95% of variance
            kappa = 0.95
            # singular values scaled to represent % of variance explained.
            S_var = S**2 / jnp.sum(S**2)
            # find index k, s.t. the first k elements in S_var account for 95% of the variance
            k_pc = jnp.argmax(jnp.cumsum(S_var) >= kappa) # returns first occurence of True
            k_pc = jnp.maximum(k_pc, jnp.ones_like(k_pc))


            # get HVPs in PCA directions
            # TODO: understnad why we use f=model/surrogate and not f=bachelier
            mtx = get_HVPs(f=MakeScalar(model), primals=x, directions=principal_components)



            

            value_loss = loss_fn(y, y_pred)
            grad_loss = loss_fn(dydx, dydx_pred)

            n_dims = x.shape[-1]
            alpha, beta = loss_balance(n_dims)

            return alpha * value_loss + beta * grad_loss







        sobolev_loss_fn = sobolev_second_order_loss

    return sobolev_loss_fn




