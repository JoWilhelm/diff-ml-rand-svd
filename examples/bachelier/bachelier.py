import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import optax

import diff_ml as dml
from diff_ml.model import Bachelier


def main():
    path = pathlib.Path(__file__).parent.absolute()
    output_folder_name = "result"
    prefix_path = f"{path}/{output_folder_name}"

    key = jrandom.PRNGKey(0)
    n_dims: int = 7
    n_samples: int = 8 * 1024

    key, subkey = jrandom.split(key)
    weights = jrandom.uniform(subkey, shape=(n_dims,), minval=1.0, maxval=10.0)

    model = Bachelier(key, n_dims, weights)

    train_ds = model.sample(n_samples)
    test_ds = model.analytic(n_samples)

    xs_train = jnp.asarray(train_ds["spot"])
    ys_train = jnp.asarray(train_ds["payoff"])
    zs_train = jnp.asarray(train_ds["differentials"])

    xs_test = jnp.asarray(test_ds["spot"])
    ys_test = jnp.asarray(test_ds["payoff"])
    zs_test = jnp.asarray(test_ds["differentials"])

    baskets = model.baskets(xs_test)

    vis_dim: int = 0
    plt.figure()
    plt.plot(xs_train[:, vis_dim], ys_train, ".", label="Payoff Training", markersize=1)
    plt.plot(baskets, ys_test, ".", label="Payoff Test", markersize=1)
    plt.legend()
    plt.savefig(f"{prefix_path}/payoff.pdf")

    plt.figure()
    plt.plot(xs_train[:, vis_dim], zs_train[:, vis_dim], ".", label="Delta Training", markersize=1)
    plt.plot(baskets, zs_test[:, vis_dim], ".", label="Delta Test", markersize=1)
    plt.legend()
    plt.savefig(f"{prefix_path}/delta.pdf")

    n_epochs = 100
    n_batch_size = 256
    train_ds = model.batch_generator(n_batch_size)

    key, subkey = jrandom.split(key)
    optim = optax.adam(learning_rate=1e-3)
    surrogate = eqx.nn.MLP(
        key=subkey, in_size=n_dims, out_size="scalar", width_size=20, depth=3, activation=jax.nn.silu
    )
    surrogate = dml.train(surrogate, train_ds, test_ds, optim, n_epochs=n_epochs)


if __name__ == "__main__":
    jax.enable_checks = True
    main()
