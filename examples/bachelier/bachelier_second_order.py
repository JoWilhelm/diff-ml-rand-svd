import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import optax
from jaxtyping import Array, Float
from jax import vmap

import diff_ml as dml
import diff_ml.nn as dnn
from diff_ml.model import Bachelier
from diff_ml.nn.utils import init_model_weights
from diff_ml.typing import Data

from diff_ml.plotting import plot_eval



def loss_fn(model, batch: Data) -> Float[Array, ""]:
    xs, ys = batch["x"], batch["y"]
    pred_ys = eqx.filter_vmap(model)(xs)
    return dml.losses.mse(ys, pred_ys)


def eval_fn(model, batch: Data) -> Float[Array, ""]:
    return jnp.sqrt(loss_fn(model, batch))


def train_generator(xs, n_samples: int, n_batch_size: int, *, key):
    while True:
        key, subkey = jrandom.split(key)

        def subset_fn(key):
            choice = jrandom.choice(key=key, a=n_samples, shape=(n_batch_size,))

            def subset(x):
                return x[choice]

            return subset

        yield jtu.tree_map(subset_fn(subkey), xs)


def main():
    # Ensure result folder exists
    path = pathlib.Path(__file__).parent.absolute()
    output_folder_name = "result"
    prefix_path = f"{path}/{output_folder_name}"
    pathlib.Path(prefix_path).mkdir(parents=True, exist_ok=True)

    # Specify model
    key = jrandom.key(0)
    n_dims: int = 7
    n_samples: int = 8 * 1024
    key, subkey = jrandom.split(key)
    weights = jrandom.uniform(subkey, shape=(n_dims,), minval=1.0, maxval=10.0)
    ref_model = Bachelier(key, n_dims, weights)



    # Generate data
    train_ds = ref_model.sample(n_samples)
    test_ds = ref_model.analytic(n_samples)




    n_epochs = 100
    n_batch_size = 256

    key, subkey = jrandom.split(key)
    train_gen = train_generator(train_ds, n_samples, n_batch_size, key=subkey)

    # Plot data
    x_train_mean = jnp.mean(train_ds["x"])
    x_train_std = jnp.std(train_ds["x"])
    y_train_mean = jnp.mean(train_ds["y"])
    y_train_std = jnp.std(train_ds["y"])

    xs_train = jnp.asarray(train_ds["x"])
    ys_train = jnp.asarray(train_ds["y"])
    zs_train = jnp.asarray(train_ds["dydx"])

    xs_test = jnp.asarray(test_ds["x"])
    ys_test = jnp.asarray(test_ds["y"])
    zs_test = jnp.asarray(test_ds["dydx"])

    baskets = ref_model.baskets(xs_test)

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
    mlp = eqx.nn.MLP(key=subkey, in_size=n_dims, out_size="scalar", width_size=20, depth=3, activation=jax.nn.silu) # jax.nn.silu

    key, subkey = jrandom.split(key)
    mlp = init_model_weights(mlp, jax.nn.initializers.glorot_normal(), key=subkey)

    surrogate = dnn.Normalized(
        dnn.Normalization(x_train_mean, x_train_std), mlp, dnn.Denormalization(y_train_mean, y_train_std)
    )

    ## Train the surrogate in the usual manner
    #optim = optax.adam(learning_rate=1e-4)
    #surrogate_std = surrogate
    #surrogate_std, metrics_std = dml.train(
    #    surrogate_std, loss_fn, train_gen, eval_fn, test_ds, optim, n_epochs=n_epochs
    #)

    # Train the surrogate using sobolev loss
    optim = optax.adam(learning_rate=1e-4)
    sobolev_loss_fn = dml.losses.sobolev(dml.losses.mse, method=dml.losses.SobolevLossType.SECOND_ORDER_PCA, ref_model=ref_model)
    surrogate, metrics = dml.train(
        surrogate, sobolev_loss_fn, train_gen, eval_fn, test_ds, optim, n_epochs=n_epochs)



    
    

    # Plot loss curve
    # plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": "EB Garamond", "font.size": 20})
    plt.figure()
    #plt.plot(jnp.sqrt(metrics_std["train_loss"]), label="Vanilla Train Loss")
    #plt.plot(metrics_std["test_loss"], label="Vanilla Test Loss")
    plt.plot(jnp.sqrt(metrics["train_loss"]), label="Sobolev Train Loss")
    plt.plot(metrics["test_loss"], label="Sobolev Test Loss")
    plt.title("Surrogates for Bachelier Basket Option")
    plt.xlabel("Epoch")
    plt.ylabel("Loss [RMSE]")
    plt.legend()
    plt.savefig(f"{prefix_path}/loss.pdf", bbox_inches="tight")










    # TODO put this somewhere where it makes more sense, separate file?

    class MakeScalar(eqx.Module):
        model: eqx.Module

        def __call__(self, *args, **kwargs):
            out = self.model(*args, **kwargs)
            return jnp.reshape(out, ())
    
    def predict(model, xs):
        pred_y, pred_dydx = vmap(eqx.filter_value_and_grad(MakeScalar(model)))(xs)
        pred_ddyddx = vmap(jax.hessian(MakeScalar(model)))(xs)

        return pred_y, pred_dydx, pred_ddyddx

    ## visualize normal predictions
    #pred_y, pred_dydx, pred_ddyddx = predict(surrogate_std, test_ds["x"])
    #plot_eval(pred_y, pred_dydx, pred_ddyddx, test_ds)

    # visualize (second order) predictions
    pred_y, pred_dydx, pred_ddyddx = predict(surrogate, test_ds["x"])
    plot_eval(pred_y, pred_dydx, pred_ddyddx, test_ds)




if __name__ == "__main__":
    jax.enable_checks = True
    main()
