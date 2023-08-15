import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import diff_ml as dml
from diff_ml.model import Bachelier


def main():
    key = jax.random.PRNGKey(0)
    n_dims: int = 1
    n_samples: int = 8 * 1024

    key, subkey = jax.random.split(key)
    weights = jax.random.uniform(subkey, shape=(n_dims,), minval=1.0, maxval=10.0)

    model = Bachelier(key, n_dims, weights)

    train_ds_gen = model.generator(n_samples)
    train_ds = next(train_ds_gen)
    test_ds = model.test_generator(n_samples)

    xs_train = jnp.asarray(train_ds["spot"])
    ys_train = jnp.asarray(train_ds["payoff"])
    zs_train = jnp.asarray(train_ds["differentials"])

    xs_test = jnp.asarray(test_ds["spot"])
    ys_test = jnp.asarray(test_ds["payoff"])
    zs_test = jnp.asarray(test_ds["differentials"])

    baskets = model.baskets(xs_test)

    vis_dim: int = 0
    plt.figure()
    plt.plot(xs_train[:, vis_dim], ys_train, "b.", label="Payoff Training", markersize=1)
    plt.plot(baskets, ys_test, "r.", label="Payoff Test", markersize=1)
    plt.legend()
    plt.savefig("output/payoff.pdf")

    plt.figure()
    plt.plot(xs_train[:, vis_dim], zs_train[:, vis_dim], "b.", label="Delta Training", markersize=1)
    plt.plot(baskets, zs_test, "r.", label="Delta Test", markersize=1)
    plt.legend()
    plt.savefig("output/delta.pdf")


if __name__ == "__main__":
    main()
