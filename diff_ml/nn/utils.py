import equinox as eqx
import jax.nn as jnn
import jax.random as jr
import jax.tree_util as jtu
from jaxtyping import PRNGKeyArray, PyTree


def is_linear(x):
    return isinstance(x, eqx.nn.Linear)


def get_weights(m: PyTree):
    return [x.weight for x in jtu.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]


def init_model_weights(model, init_fn: jnn.initializers.Initializer, *, key: PRNGKeyArray):
    weights = get_weights(model)
    new_weights = [init_fn(subkey, weight.shape) for weight, subkey in zip(weights, jr.split(key, len(weights)))]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model
