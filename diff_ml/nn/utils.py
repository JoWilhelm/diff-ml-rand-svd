import jax.numpy as jnp
import equinox as eqx
import jax.nn as jnn
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import Float, Array, PRNGKeyArray, PyTree
from jax import vmap, hessian


class LossState(eqx.Module):
    losses: Float[Array, "n_losses"] = jnp.ones(3) * 1/3
    lambdas: Float[Array, "n_losses"] = jnp.ones(3) * 1/3
    initial_losses: Float[Array, "n_losses"] = jnp.ones(3) * 1/3
    accum_losses: Float[Array, "n_losses"] = jnp.zeros(3)
    prev_mean_losses: Float[Array, "n_losses"] = jnp.zeros(3)
    current_iter: Float[Array, "n_losses"] = jnp.zeros(3)

    def update_losses(state, new_value: Float[Array, "n_losses"]): # -> LossState:
        return LossState(new_value, state.lambdas, state.initial_losses, state.accum_losses, state.prev_mean_losses, state.current_iter)

    def update_lambdas(state, new_value: Float[Array, "n_losses"]):# -> LossState:
        return LossState(state.losses, new_value, state.initial_losses, state.accum_losses, state.prev_mean_losses,  state.current_iter)
        
    def update_initial_losses(state, new_value: Float[Array, "n_losses"]): # -> LossState:
        return LossState(state.losses, state.lambdas, new_value, state.accum_losses, state.prev_mean_losses,  state.current_iter)
        
    def update_accum_losses(state, new_value: Float[Array, "n_losses"]): # -> LossState:
        return LossState(state.losses, state.lambdas, state.initial_losses, new_value, state.prev_mean_losses,  state.current_iter)
        
    def update_current_iter(state, new_value: Float[Array, "n_losses"]): # -> LossState:
        return LossState(state.losses, state.lambdas, state.initial_losses, state.accum_losses, state.prev_mean_losses,  new_value)

    def update_prev_mean_losses(state, new_value: Float[Array, "n_losses"]): # -> LossState:
        return LossState(state.losses, state.lambdas, state.initial_losses, state.accum_losses, new_value,  state.current_iter)

    def __repr__(self):
        return "LossState"
    


class MakeScalar(eqx.Module):
    model: eqx.Module

    def __call__(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        return jnp.reshape(out, ())




def is_linear(x):
    return isinstance(x, eqx.nn.Linear)


def get_weights(m: PyTree):
    return [x.weight for x in jtu.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]


def init_model_weights(model, init_fn: jnn.initializers.Initializer, *, key: PRNGKeyArray):
    weights = get_weights(model)
    new_weights = [init_fn(subkey, weight.shape) for weight, subkey in zip(weights, jr.split(key, len(weights)))]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model


    
def predict(model, xs):
    pred_y, pred_dydx = vmap(eqx.filter_value_and_grad(MakeScalar(model)))(xs)
    pred_ddyddx = vmap(hessian(MakeScalar(model)))(xs)

    return pred_y, pred_dydx, pred_ddyddx
