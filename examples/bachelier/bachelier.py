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


def trunc_init(weight: Array, key: PRNGKeyArray) -> Array:
    out, in_ = weight.shape
    # stddev = jnp.sqrt(1 / in_)
    return jax.nn.initializers.glorot_normal()(key, (out, in_))


def init_linear_weight(model, init_fn, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)]
    weights = get_weights(model)
    new_weights = [init_fn(weight, subkey) for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model


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
    # for k,v in train_ds.items():
    #         print(k, v.shape)

    test_ds = model.analytic(n_samples)

    n_epochs = 100
    n_batch_size = 256
    train_gen = model.batch_generator(n_batch_size)

    # train_ds = next(train_gen)
    # train_ds = next(train_gen)
    # train_ds = next(train_gen)
    # train_ds = next(train_gen)
    # train_ds = next(train_gen)
    # train_ds = next(train_gen)
    #

    # print("diff ", train_ds1["spot"])
    # print("diff ", train_ds2["spot"])

    # train_ds = repeat(train_ds, n_epochs)
    # for _ in range(n_epochs):
    #     train_i = next(train_gen)
    # train_ds |= train_i # union

    # for k,v in train_ds.items():
    #     print(k, v.shape)

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

    # baskets = model.baskets(xs_test)

    # vis_dim: int = 0
    # plt.figure()
    # plt.plot(xs_train[:, vis_dim], ys_train, ".", label="Payoff Training", markersize=1)
    # plt.plot(baskets, ys_test, ".", label="Payoff Test", markersize=1)
    # plt.legend()
    # plt.savefig(f"{prefix_path}/payoff.pdf")
    #
    # plt.figure()
    # plt.plot(xs_train[:, vis_dim], zs_train[:, vis_dim], ".", label="Delta Training", markersize=1)
    # plt.plot(baskets, zs_test[:, vis_dim], ".", label="Delta Test", markersize=1)
    # plt.legend()
    # plt.savefig(f"{prefix_path}/delta.pdf")

    if 0:
        optim = optax.adam(learning_rate=1e-4)

        # s = dml.nn.Normalized
        # s = dml.nn.Normalized

        # surrogate = eqx.nn.MLP(
        #     key=subkey, in_size=n_dims, out_size="scalar", width_size=20, depth=3, activation=jax.nn.silu
        # )
        # surrogate = dml.train(surrogate, train_ds, test_ds, optim, n_epochs=n_epochs)

        key, subkey = jrandom.split(key)
        mlp = eqx.nn.MLP(key=subkey, in_size=n_dims, out_size="scalar", width_size=20, depth=3, activation=jax.nn.silu)

        key, subkey = jrandom.split(key)
        mlp = init_linear_weight(mlp, trunc_init, subkey)

        surrogate = dml.Normalized(
            dml.Normalization(x_train_mean, x_train_std), mlp, dml.Denormalization(y_train_mean, y_train_std)
        )

        surrogate = dml.train(surrogate, train_gen, test_ds, optim, n_epochs=n_epochs)

    else:
        print("Alternative branch")
        optim = optax.adam(learning_rate=1e-4)
        # surrogate = dml.nn.Normalized(dml.nn.Normalization(), model, dml.nn.Denormalization())
        # surrogate = dnn.Normalized

        def x_normalizer(x):
            return (x - x_train_mean) / x_train_std

        def y_denormalizer(x):
            return x * y_train_std + y_train_mean

        class MLP_Normalized(eqx.Module):
            layers: list
            in_size: Union[int, Literal["scalar"]] = eqx.field(static=True)
            out_size: Union[int, Literal["scalar"]] = eqx.field(static=True)

            def __init__(self, key, in_size, out_size):
                key, subkey = jax.random.split(key)
                self.layers = [
                    x_normalizer,
                    eqx.nn.MLP(
                        key=subkey,
                        in_size=in_size,
                        out_size=out_size,
                        width_size=20,
                        depth=3,
                        activation=jax.nn.silu,
                    ),
                    y_denormalizer,
                ]
                self.in_size = in_size
                self.out_size = out_size

            def __call__(self, x):
                for layer in self.layers:
                    x = layer(x)

                return x

        key, subkey = jrandom.split(key)
        surrogate = MLP_Normalized(key, in_size=x_train.shape[1], out_size="scalar")

        # print("weights of first layer are random")
        # first_weights = surrogate.layers[1].layers[1].weight
        # print(jnp.mean(first_weights), jnp.std(first_weights))

        key, subkey = jax.random.split(key)
        surrogate = init_linear_weight(surrogate, trunc_init, key)

        # print("weights of first layer are now glorot random")
        # glorot_weights = surrogate.layers[1].layers[1].weight
        # print(jnp.mean(glorot_weights), jnp.std(glorot_weights))

        surrogate = dml.train(surrogate, train_gen, test_ds, optim, n_epochs=n_epochs)



if __name__ == "__main__":
    jax.enable_checks = True
    main()
