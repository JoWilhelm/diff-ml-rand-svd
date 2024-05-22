import pathlib
from typing import Literal, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import optax
from jaxtyping import Array, PRNGKeyArray

import diff_ml as dml
from diff_ml.model import Bachelier


def normalize(data: Array):
    mean = jnp.mean(data)
    std = jnp.std(data)
    return (data - mean) / std, mean, std


# def init_layer_weights(weight: Array, key: PRNGKeyArray) -> Array:
#     return jax.nn.initializers.glorot_normal()(key, weight.shape)


def init_model_weights(model, init_fn, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
    weights = get_weights(model)
    new_weights = [init_fn(subkey, weight.shape) for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model


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

    key = jrandom.PRNGKey(0)
    n_dims: int = 7
    n_samples: int = 8 * 1024

    key, subkey = jrandom.split(key)
    weights = jrandom.uniform(subkey, shape=(n_dims,), minval=1.0, maxval=10.0)

    model = Bachelier(key, n_dims, weights)

    train_ds = model.sample(n_samples)
    test_ds = model.analytic(n_samples)

    n_epochs = 100
    n_batch_size = 256
    # train_gen = model.batch_generator(n_batch_size)
    # print(next(train_gen))
    key, subkey = jrandom.split(key)
    train_gen = generator_from_samples(train_ds, n_samples, n_batch_size, key=subkey)

    x_train = jnp.asarray(train_ds["spot"])
    y_train = jnp.asarray(train_ds["payoff"])
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

    key, subkey = jrandom.split(key)
    mlp = eqx.nn.MLP(key=subkey, in_size=n_dims, out_size="scalar", width_size=20, depth=3, activation=jax.nn.silu)

    key, subkey = jrandom.split(key)
    mlp = init_model_weights(mlp, jax.nn.initializers.glorot_normal(), subkey)
#     return jax.nn.initializers.glorot_normal()(key, weight.shape)

    surrogate = dml.Normalized(
        dml.Normalization(x_train_mean, x_train_std), mlp, dml.Denormalization(y_train_mean, y_train_std)
    )

    optim = optax.adam(learning_rate=1e-4)
    surrogate = dml.train(surrogate, train_gen, test_ds, optim, n_epochs=n_epochs)


if __name__ == "__main__":
    jax.enable_checks = True
    main()
