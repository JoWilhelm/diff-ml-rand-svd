from itertools import islice
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import optax
from jax import vmap
from jaxtyping import Array, Float, Int, PyTree

from diff_ml import Data, DataGenerator
from diff_ml.nn import loss
from diff_ml.nn.modules import MakeScalar


# from diff_ml.nn.loss import mse, rmse

# @eqx.filter_jit
# def loss_fn(
#     model: PyTree, x: Float[Array, " batch"], y: Float[Array, " batch"]
# ) -> Float:
#     pred_y = vmap(MakeScalar(model))(x)
#     pred_y = pred_y[:, jnp.newaxis]
#     result = mse(y, pred_y)
#     return result
#

@eqx.filter_jit
def loss_fn(
    model, x, y
):
    pred_y = vmap(model)(x)
    pred_y = pred_y[:, jnp.newaxis]
    result = loss.mse(y, pred_y)
    return result


def evaluate(model, testloader):
    """This function evaluates the model on the test dataset, computing both the average loss."""
    avg_loss = 0
    for x, y, _ in testloader.values():
        avg_loss += loss_fn(model, x, y)
    return avg_loss / len(testloader["spot"])


def train(
    model: PyTree,
    train_data: DataGenerator,
    # test_data: Optional[DataGenerator],
    test_data: Optional[Data],
    optim: optax.GradientTransformation,
    n_epochs: int,
) -> PyTree:
    @eqx.filter_jit
    def train_step(model: PyTree, opt_state: PyTree, batch: Data):
        x, y, _ = batch.values()
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    train_loss = jnp.zeros(1)

    for epoch in range(n_epochs):
        for batch in islice(train_data, 3):
            model, opt_state, train_loss = train_step(model, opt_state, batch)

        epoch_stats = f"Finished epoch {epoch:3d} | Train Loss: {train_loss:.5f}"

        if test_data:
            test_loss = jnp.sqrt(loss_fn(model, test_data["spot"], test_data["payoff"]))
            print(f"{epoch_stats} | Test Loss: {test_loss:.5f}")
        else:
            print(epoch_stats)

    return model
