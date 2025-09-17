import jax
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray

# Utility functions for model initialization
# Credit: Neil Kichler 

def trunc_init(weight: Array, key: PRNGKeyArray) -> Array:
  out, in_ = weight.shape
  return jax.nn.initializers.glorot_normal()(key, (out, in_))


def init_linear_weight(model, init_fn, key):
  is_linear = lambda x: isinstance(x, eqx.nn.Linear)
  get_weights = lambda m: [x.weight
                           for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                           if is_linear(x)]
  weights = get_weights(model)
  new_weights = [init_fn(weight, subkey)
                 for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
  new_model = eqx.tree_at(get_weights, model, new_weights)
  return new_model