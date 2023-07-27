import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array

import diff_ml as dml
from diff_ml.model import Bachelier, BachelierParams, generate_correlation_matrix


class TestGenerateCorrelatedSamples:
    def test_generate_correlated_samples(self):
        key = jrandom.PRNGKey(0)
        n_samples = 1024
        samples = generate_correlation_matrix(key, n_samples)
        assert samples.shape == (n_samples, n_samples)


    def test_generator(self):
        key = jrandom.PRNGKey(0)
        bachelier = Bachelier(key, n_dims=7)
        spots_1, payoffs_vjp, differentials_analytic, differentials_vjp = bachelier.generator(1024)
        assert jnp.allclose(differentials_analytic, differentials_vjp)
