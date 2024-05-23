import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import optax
from jaxtyping import Array, Float

import diff_ml as dml
from diff_ml.model import Bachelier
from diff_ml.nn.utils import init_model_weights


def loss_fn(model, batch: dml.Data) -> Float[Array, ""]:
    xs, ys = batch["spot"], batch["payoff"]
    pred_ys = eqx.filter_vmap(model)(xs)
    return dml.losses.mse(ys, pred_ys)


def generator_from_samples(xs, n_samples: int, n_batch_size: int, *, key):
    while True:
        key, subkey = jrandom.split(key)
        choice = jrandom.choice(key=subkey, a=n_samples, shape=(n_batch_size,))
        yield jax.tree_util.tree_map(lambda x: x[choice], xs)


def main():
    path = pathlib.Path(__file__).parent.absolute()
    output_folder_name = "result"
    prefix_path = f"{path}/{output_folder_name}"
    pathlib.Path(prefix_path).mkdir(parents=True, exist_ok=True)

    key = jrandom.key(0)
    n_dims: int = 7
    n_samples: int = 8 * 1024

    key, subkey = jrandom.split(key)
    weights = jrandom.uniform(subkey, shape=(n_dims,), minval=1.0, maxval=10.0)

    model = Bachelier(key, n_dims, weights)

    train_ds = model.sample(n_samples)
    test_ds = model.analytic(n_samples)

    n_epochs = 100
    n_batch_size = 256

    key, subkey = jrandom.split(key)
    train_gen = generator_from_samples(train_ds, n_samples, n_batch_size, key=subkey)
    # Alternatively use the batch generator
    # train_gen = model.batch_generator(n_batch_size)

    x_train_mean = jnp.mean(train_ds["spot"])
    x_train_std = jnp.std(train_ds["spot"])
    y_train_mean = jnp.mean(train_ds["payoff"])
    y_train_std = jnp.std(train_ds["payoff"])

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

    # Specify the surrogate model architecture
    key, subkey = jrandom.split(key)
    mlp = eqx.nn.MLP(key=subkey, in_size=n_dims, out_size="scalar", width_size=20, depth=3, activation=jax.nn.silu)

    key, subkey = jrandom.split(key)
    mlp = init_model_weights(mlp, jax.nn.initializers.glorot_normal(), key=subkey)

    surrogate = dml.Normalized(
        dml.Normalization(x_train_mean, x_train_std), mlp, dml.Denormalization(y_train_mean, y_train_std)
    )

    # Train the surrogate
    optim = optax.adam(learning_rate=1e-4)
    surrogate = dml.train(surrogate, loss_fn, train_gen, test_ds, optim, n_epochs=n_epochs)


if __name__ == "__main__":
    jax.enable_checks = True
    main()
