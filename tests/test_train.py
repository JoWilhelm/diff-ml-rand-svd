import typing
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import optax
import tensorflow_datasets as tfds
from jaxtyping import Array, PRNGKeyArray

import diff_ml as dml
import diff_ml.nn as dnn
from datasets import Dataset, DatasetInfo, load_from_disk
from diff_ml import Data, DataGenerator
from diff_ml.model import Bachelier, generate_correlation_matrix


def trunc_init(weight: Array, key: PRNGKeyArray) -> Array:
    out, in_ = weight.shape
    jnp.sqrt(1 / in_)
    return jax.nn.initializers.glorot_normal()(key, (out, in_))


def init_linear_weight(model, init_fn, key):
    def is_linear(x):
        return isinstance(x, eqx.nn.Linear)

    def get_weights(m):
        return [
            x.weight
            for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
            if is_linear(x)
        ]

    weights = get_weights(model)
    new_weights = [
        init_fn(weight, subkey)
        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    ]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model


class TestTrain:
    def test_train(self):
        key = jrandom.PRNGKey(0)
        n_dims: int = 1
        n_precompute = 1024
        n_samples: int = 8 * 1024
        n_epochs: int = 3
        n_batch_size: int = 512

        key, subkey = jrandom.split(key)
        weights = jrandom.uniform(subkey, shape=(n_dims,), minval=1.0, maxval=10.0)

        model = Bachelier(key, n_dims, weights)
        partial(model.generator, n_precompute=n_precompute)
        train_ds: Data = model.sample(n_samples)
        test_ds: Data = model.analytic(n_samples)

        x_mean = jnp.mean(train_ds["spot"])
        x_std = jnp.std(train_ds["spot"])

        y_mean = jnp.mean(train_ds["payoff"])
        y_std = jnp.std(train_ds["payoff"])

        key, subkey = jrandom.split(key)
        mlp = eqx.nn.MLP(
            key=subkey,
            in_size=n_dims,
            out_size=1,
            width_size=20,
            depth=3,
            activation=jax.nn.elu,
        )

        x_normalizer = dnn.Normalization(mean=x_mean, std=x_std)
        y_denormalizer = dnn.Denormalization(mean=y_mean, std=y_std)
        surrogate = dnn.Normalized(x_normalizer, mlp, y_denormalizer)
        key, subkey = jrandom.split(key)
        surrogate = init_linear_weight(surrogate, trunc_init, subkey)

        total_steps = n_epochs * (len(train_ds) // n_batch_size) + n_epochs

        lr_schedule = optax.exponential_decay(
            init_value=0.001,
            transition_steps=total_steps,
            transition_begin=int(total_steps * 0.2),
            decay_rate=0.9,
        )
        optim = optax.adam(learning_rate=lr_schedule)
        surrogate = dnn.train(
            surrogate,
            model.batch_generator(n_precompute),
            test_ds,
            optim,
            n_epochs=n_epochs,
        )

        # jnp.asarray(train_ds["spot"])
        # jnp.asarray(train_ds["payoff"])
        # jnp.asarray(train_ds["differentials"])
        #
        # xs_test = jnp.asarray(test_ds["spot"])
        # jnp.asarray(test_ds["payoff"])
        # jnp.asarray(test_ds["differentials"])
        #
        # model.baskets(xs_test)


if __name__ == "__main__":
    TestTrain().test_train()
