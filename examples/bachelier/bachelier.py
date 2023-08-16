import pathlib

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from diff_ml.model import Bachelier


def main():
    path = pathlib.Path(__file__).parent.absolute()
    output_folder_name = "result"
    prefix_path = f"{path}/{output_folder_name}"

    key = jax.random.PRNGKey(0)
    n_dims: int = 1
    n_samples: int = 8 * 1024

    key, subkey = jax.random.split(key)
    weights = jax.random.uniform(subkey, shape=(n_dims,), minval=1.0, maxval=10.0)

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
    plt.plot(xs_train[:, vis_dim], ys_train, "b.", label="Payoff Training", markersize=1)
    plt.plot(baskets, ys_test, "r.", label="Payoff Test", markersize=1)
    plt.legend()
    plt.savefig(f"{prefix_path}/payoff.pdf")

    plt.figure()
    plt.plot(xs_train[:, vis_dim], zs_train[:, vis_dim], "b.", label="Delta Training", markersize=1)
    plt.plot(baskets, zs_test, "r.", label="Delta Test", markersize=1)
    plt.legend()
    plt.savefig(f"{prefix_path}/delta.pdf")


if __name__ == "__main__":
    main()
