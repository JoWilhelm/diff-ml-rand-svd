from collections.abc import Callable
from enum import Enum

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float

from functools import partial

import diff_ml as dml
from diff_ml.hvp_stuff import hvp_batch, cfd
from diff_ml.model.bachelier import Bachelier


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



def normalize_vectors(vectors):
    return vectors / jnp.linalg.norm(vectors, axis=1, keepdims=True)

def generate_random_vectors(key, k, dim, normalize=True):
    key, subkey = jax.random.split(key)
    vectors = jax.random.normal(subkey, shape=(k, dim))
    if normalize:
        vectors = normalize_vectors(vectors)
    return vectors


# TODO separate first order and second order loss functions
@jax.named_scope("dml.losses.sobolev")
def sobolev(loss_fn: RegressionLossFn, *, method: SobolevLossType = SobolevLossType.FIRST_ORDER, ref_model) -> RegressionLossFn:
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

        def loss_balance() -> tuple[float, float, float]:
            return 1/3, 1/3, 1/3



        def sobolev_second_order_loss(model, batch, ref_model=ref_model) -> Float[Array, ""]:

            # unpack dictionary for readability
            x, y, dydx, paths1 = batch["x"], batch["y"], batch["dydx"], batch["paths1"]
            
    
            # get surrogate prediction, first-order derivative and hessian
            y_pred, dydx_pred = vmap(eqx.filter_value_and_grad(MakeScalar(model)))(x)
            hessian = eqx.filter_vmap(jax.hessian(MakeScalar(model)))(x) # TODO move lower where needed?
            assert y.shape == y_pred.shape
            assert dydx.shape == dydx_pred.shape

            # loss and first-order differntial loss
            value_loss = loss_fn(y, y_pred)
            grad_loss = loss_fn(dydx, dydx_pred)



            # get directions for hessian probing for second-order loss

            # in random directions
            key = jax.random.key(42)
            rand_directions = generate_random_vectors(key, k=7, dim=x.shape[-1])
            

            # TODO PCA stuff
            # apply PCA to first-order gradients of predictions
            dydx_used = dydx_pred
            #dydx_used = dydx               # alternatively use dydx of reference model
            #dydx_used = dydx_pred - dydx   # ore difference between the two
            dydx_means = jnp.mean(dydx_used, axis=0)
            tiled_dydx_used_means = jnp.tile(dydx_means, (dydx_used.shape[0], 1))
            dydx_used_mean_adjusted = dydx_used - tiled_dydx_used_means
            U, S, VT = jnp.linalg.svd(dydx_used_mean_adjusted, full_matrices=False)
            principal_components = jnp.diag(S) @ VT
            
            #jax.debug.print("principal_components.shape {shape}", shape=principal_components.shape)
            #jax.debug.print("principal_components[0] {pc0}", pc0=principal_components[0])
            #jax.debug.print("")
            #return .0

            pca_directions = normalize_vectors(principal_components.T)


            # select PCs that account for 95% of variance
            kappa = 0.95
            # singular values scaled to represent % of variance explained.
            S_var = S**2 / jnp.sum(S**2)
            compute_hvp_ = ~(jnp.cumsum(S_var) > kappa)
            compute_hvp = compute_hvp_.at[0].set(True) # make use that at least the first principal component is always actively used.
            

            #jax.debug.print("compute_hvp {v}", v=compute_hvp)
            #jax.debug.print("")
            #return .0

            ## get HVPs of surrogate in PCA directions
            ## TODO: understnad why we use f=model/surrogate and not f=bachelier
            #mtx = get_HVPs(f=MakeScalar(model), primals=x, directions=principal_components)






            # ----
            

            # generate second-order targets via finite differences
            
            #directions = pca_directions
            directions = rand_directions

            payoff_fn = partial(ref_model.antithetic_payoff, # TODO make loss function independent of Bachelier, pass payoff_fn
                                    weights=ref_model.weights,
                                    strike_price=ref_model.strike_price
                                )
            D_payoff_fn = jax.vmap(jax.grad(payoff_fn)) # TODO understand all these vmaps
            h = 1e-1
            cfd_of_dpayoff_fn = cfd(D_payoff_fn, h, x, paths1) # TODO inc1 = paths1 understand what this is and where it comes from
            ddpayoff = jax.vmap(cfd_of_dpayoff_fn)(directions) 
            ddpayoff = jnp.transpose(ddpayoff, (1, 0, 2))
            #jax.debug.print("directions.shape {shape}", shape=directions.shape)
            #jax.debug.print("paths1.shape {shape}", shape=paths1.shape)
            #jax.debug.print("x.shape {shape}", shape=x.shape)
            #jax.debug.print("")
    





            # second order predictions

            hvps_pred = hvp_batch(f=MakeScalar(model), 
                                         inputs=x, 
                                         directions=directions
                                         )
            #jax.debug.print("rand_directions.shape {shape}", shape=rand_directions.shape)
            #jax.debug.print("hvps_rand.shape {shape}", shape=hvps_rand.shape)
            #jax.debug.print("")







            hessian_loss = loss_fn(ddpayoff, hvps_pred)

            
            alpha, beta, gamma = loss_balance()
            
            loss = alpha * value_loss + beta * grad_loss + gamma * hessian_loss



            return loss
    







        sobolev_loss_fn = sobolev_second_order_loss

    return sobolev_loss_fn









