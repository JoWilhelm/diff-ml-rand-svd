import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array

import diff_ml as dml
from diff_ml.model import Bachelier, BachelierParams, generate_correlated_samples


class TestGenerateCorrelatedSamples:
    def test_generate_correlated_samples(self):
        key = jrandom.PRNGKey(0)
        n_samples = 1024
        samples = generate_correlated_samples(key, n_samples)
        assert samples.shape == (n_samples, n_samples)
