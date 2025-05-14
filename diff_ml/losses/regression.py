from collections.abc import Callable
from enum import Enum

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float

from functools import partial

import diff_ml as dml
from diff_ml.hvp_stuff import hvp_batch, hvp_batch_cond, cfd_fn, cfd_cond_fn
from diff_ml.model.bachelier import Bachelier

from diff_ml.nn.utils  import LossState


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

def generate_random_vectors(k, dim, key, normalize=True):
    key, subkey = jax.random.split(key)
    vectors = jax.random.normal(subkey, shape=(k, dim))
    if normalize:
        vectors = normalize_vectors(vectors)
    return vectors



# apply PCA to first-order gradients of predictions
def PCA_of_dydx_directions(dydx, kappa=0.95, normalize=True):
    
    
    dydx_means = jnp.mean(dydx, axis=0)
    tiled_dydx_used_means = jnp.tile(dydx_means, (dydx.shape[0], 1))
    dydx_used_mean_adjusted = dydx - tiled_dydx_used_means
    U, S, VT = jnp.linalg.svd(dydx_used_mean_adjusted, full_matrices=False)
    principal_components = jnp.diag(S) @ VT
    pca_directions = principal_components.T
    #jax.debug.print("principal_components.shape {shape}", shape=principal_components.shape)
    #jax.debug.print("principal_components[0] {pc0}", pc0=principal_components[0])
    #jax.debug.print("")
    #return .0

    if normalize:
        pca_directions = normalize_vectors(pca_directions)


    # select PCs that account for kappa% of variance
    # singular values scaled to represent % of variance explained.
    S_var = S**2 / jnp.sum(S**2)
    eval_hvp = (~(jnp.cumsum(S_var) > kappa)).at[0].set(True) # make use that at least the first principal component is always actively used
    k_pc = jnp.sum(eval_hvp) # number of principal components used
    
    #jax.debug.print("eval_hvp {v}", v=eval_hvp)
    #jax.debug.print("")
    #return .0

    return pca_directions, eval_hvp, k_pc










# TODO separate first order and second order loss functions?
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
        


        def loss_balance_even() -> tuple[float, float, float]:

            return 1/3, 1/3, 1/3


        def pca_loss_balance(prev_loss_state, new_loss_state, k_pc, n_dims):  
            
            lam = 1
            
            # NOTE: An appropriate eta is crucial for a working second-order method
            eta = (k_pc / n_dims) ** 2
            scale = (1 + lam * n_dims + eta * n_dims * n_dims)
            alpha = 1 / scale
            beta = (lam * n_dims) / scale
            gamma = 1.0 - alpha - beta


           
            use_balance = True
            if use_balance:
                # TODO: The loss weights should be based on the previous _weighted_ losses. The losses themselves don't 
                #       have to be balanced, only after the weighting there should be balance.
                #
                #       Maybe combine: - prev. weighted loss
                #                      - diff between current and prev loss
                #                      - maybe prev. lambdas?
                #  to predict the lambdas needed to make all the _weighted_ losses of the current step equal.


                # loss_weights = current_mean_losses / jnp.sum(current_mean_losses)
                # jax.debug.print("[{}] loss_weights: {}", current_iter, loss_weights)


                n_steps = 16.0
                is_last_step = new_loss_state.current_iter[0] == n_steps


                # NOTE: raw weighting s.t. the losses are equal does not work as it neglects larger losses in favor of smaller losses
                loss_weights = new_loss_state.losses / jnp.sum(new_loss_state.losses)
                loss_balance = 1/3 / loss_weights
                weighted_loss = jnp.multiply(loss_balance, new_loss_state.losses)
                weighting = loss_balance / jnp.sum(loss_balance)
                balancing = jnp.ones(len(new_loss_state.losses)) - loss_weights
                softmax_balancing = jax.nn.softmax(balancing)

                updated_balance_weight = 10
                softmax_balancing = updated_balance_weight * softmax_balancing
                # lambdas = jnp.array([alpha, beta, gamma])
                # lambdas_new_weights = jax.nn.softmax(lambdas + softmax_balancing)
                # jax.debug.print("lambdas new weights: {}", lambdas_new_weights)
                # alpha, beta, gamma = lambdas_new_weights

                new_lambdas = jax.nn.softmax(jnp.multiply(softmax_balancing, new_loss_state.lambdas)) * is_last_step + (1.0 - is_last_step) * new_loss_state.lambdas
                new_loss_state = new_loss_state.update_lambdas(new_lambdas)
                #alpha, beta, gamma = new_lambdas

                alpha, beta, gamma = weighting


            return alpha, beta, gamma, new_loss_state











        def sobolev_second_order_loss(model, batch, prev_loss_state, ref_model=ref_model) -> Float[Array, ""]:

            # unpack dictionary for readability
            x, y, dydx, paths1 = batch["x"], batch["y"], batch["dydx"], batch["paths1"]
            
            # get surrogate prediction, first-order derivative and hessian
            y_pred, dydx_pred = vmap(eqx.filter_value_and_grad(MakeScalar(model)))(x)
            assert y.shape == y_pred.shape
            assert dydx.shape == dydx_pred.shape

            # loss and first-order differntial loss
            value_loss = loss_fn(y, y_pred)
            grad_loss = loss_fn(dydx, dydx_pred)




            
            #### ---- Get Directions for Hessian Probing ---- ####


            # in random directions
            rand_directions = generate_random_vectors(k=7, dim=x.shape[-1], key=jax.random.key(42))
            
            # apply PCA to first-order gradients dydx_pred
            # alternatively use dydx of reference model or difference: dydx_pred - dydx
            pca_directions, eval_hvp, k_pc = PCA_of_dydx_directions(dydx_pred)
    
        



            #### ---- Second-Order Targets via Finite Differences ---- ####
            

            payoff_fn = partial(ref_model.antithetic_payoff, # TODO make loss function independent of Bachelier, pass payoff_fn
                                weights=ref_model.weights,
                                strike_price=ref_model.strike_price
                                )
            D_payoff_fn = jax.vmap(jax.grad(payoff_fn)) 
            

            # central finite differences derivative
            h = 1e-1
            cfd_of_dpayoff_fn = cfd_fn(D_payoff_fn, h, x, paths1) 
            

            
            directions = pca_directions
            #directions = rand_directions

            # all directions 
            ddpayoff = jax.vmap(cfd_of_dpayoff_fn)(directions) 
            ddpayoff = jnp.transpose(ddpayoff, (1, 0, 2))
            #jax.debug.print("ddpayoff[{i}] {v}", i=i, v=ddpayoff[i])
            
            # conditional directions
            cfd_of_dpayoff_cond_fn = cfd_cond_fn(cfd_of_dpayoff_fn, batch_size=x.shape[0]) # TODO get rid of the explicit batch size dependency
            ddpayoff_cond = cfd_of_dpayoff_cond_fn(directions, eval_hvp)
            ddpayoff_cond = jnp.transpose(ddpayoff_cond, (1, 0, 2))
            #jax.debug.print("ddpayoff_cond[{i}] {v}", i=i, v=ddpayoff_cond[i]) 






            #### ---- Second-Order Predicitons via HVPs ---- ####


            # all directions
            hvps_pred = hvp_batch(f=MakeScalar(model), 
                                         inputs=x, 
                                         directions=directions
                                         )
            #jax.debug.print("directions.shape {shape}", shape=directions.shape)
            #jax.debug.print("hvps_pred.shape {shape}", shape=hvps_pred.shape)
            #jax.debug.print("hvps_pred[{i}][0] {v}", i=i, v=hvps_pred[i][0])
            #jax.debug.print("")
            # return .0

            # conditional directions
            hvps_cond_pred = hvp_batch_cond(f=MakeScalar(model), 
                                         inputs=x, 
                                         directions=directions,
                                         eval_hvp=eval_hvp
                                         )
            #jax.debug.print("directions.shape {shape}", shape=directions.shape)
            #jax.debug.print("compute_hvp {v}", v=compute_hvp)
            #jax.debug.print("hvps_cond_pred.shape {shape}", shape=hvps_cond_pred.shape)
            #jax.debug.print("hvps_cond_pred[{i}][0] {v}", i=i, v=hvps_cond_pred[i][0])
            #jax.debug.print("")
            #return .0





            #### ---- Loss and Loss Balancing ---- ####


            #hessian_loss = loss_fn(ddpayoff, hvps_pred)
            hessian_loss = loss_fn(ddpayoff_cond, hvps_cond_pred)

            
            losses = jnp.array([value_loss, grad_loss, hessian_loss])
            new_loss_state = LossState(losses, prev_loss_state.lambdas, prev_loss_state.initial_losses, prev_loss_state.accum_losses + losses, prev_loss_state.prev_mean_losses, prev_loss_state.current_iter + 1.0)

            #alpha, beta, gamma = loss_balance_even()
            alpha, beta, gamma, new_loss_state = pca_loss_balance(prev_loss_state, new_loss_state, k_pc, x.shape[-1])
            #jax.debug.print("alpha: {a:.2f}, beta: {b:.2f}, gamma: {c:.2f}", a=alpha, b=beta, c=gamma)
            
            
            loss = alpha * value_loss + beta * grad_loss + gamma * hessian_loss
            
            return (loss, new_loss_state)
            # TODO also return alpha, beta, gamma for plotting over epochs?

    

        sobolev_loss_fn = sobolev_second_order_loss

    return sobolev_loss_fn









