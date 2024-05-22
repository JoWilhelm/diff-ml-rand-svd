from itertools import islice
from typing import Callable, Optional
from typing_extensions import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, PyTree

from diff_ml import Data, DataGenerator
from diff_ml.nn import loss


Model: TypeAlias = Callable[[PyTree], PyTree]


def loss_fn(model: Model, x: Float[Array, "batch ..."], y: Float[Array, " batch"]) -> Float:
    pred_y = eqx.filter_vmap(model)(x)
    result = loss.mse(y, pred_y)
    return result


def evaluate(model: Model, testloader, loss_fn):
    """This function evaluates the model on the test dataset, computing the average loss."""
    avg_loss = 0
    for x, y, _ in testloader.values():
        avg_loss += loss_fn(model, x, y)
    return avg_loss / len(testloader["spot"])


def train(
    model: Model,
    train_data: DataGenerator,
    # test_data: Optional[DataGenerator],
    test_data: Optional[Data],
    optim: optax.GradientTransformation,
    n_epochs: int,
) -> PyTree:
    @eqx.filter_jit
    def train_step(model, opt_state: PyTree, batch: Data):
        xs, ys = batch["spot"], batch["payoff"]
        loss_values, grads = eqx.filter_value_and_grad(loss_fn)(model, xs, ys)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_values

    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    train_loss = jnp.zeros(1)

    for epoch in range(n_epochs):
        for batch in islice(train_data, 64):  # number of batches per epoch
            model, opt_state, train_loss = train_step(model, opt_state, batch)

        epoch_stats = f"Finished epoch {epoch:3d} | Train Loss: {train_loss:.5f}"

        if test_data:
            test_loss = jnp.sqrt(loss_fn(model, test_data["spot"], test_data["payoff"]))
            print(f"{epoch_stats} | Test Loss: {test_loss:.5f}")
        else:
            print(epoch_stats)

    return model
