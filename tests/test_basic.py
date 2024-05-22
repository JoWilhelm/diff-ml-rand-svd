import jax.numpy as jnp

import diff_ml as dml


def test_basic():
    assert dml.mse(jnp.zeros((8,)), jnp.zeros((8,))) == 0.0
