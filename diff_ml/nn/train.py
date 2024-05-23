from itertools import islice
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import optax
from jaxtyping import PyTree
from tqdm import tqdm

from diff_ml import Data, DataGenerator


def evaluate(model, testloader, loss_fn):
    """This function evaluates the model on the test dataset, computing the average loss."""
    avg_loss = 0
    for x, y, _ in testloader.values():
        avg_loss += loss_fn(model, x, y)
    return avg_loss / len(testloader["spot"])


@eqx.filter_jit
def train_step(model, loss_fn, optim: optax.GradientTransformation, opt_state: PyTree, batch: Data):
    loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


def train(
    model,
    loss_fn,
    train_data: DataGenerator,
    test_data: Optional[Data],
    optim: optax.GradientTransformation,
    n_epochs: int,
) -> PyTree:
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    train_loss = jnp.zeros(1)

    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        for batch in islice(train_data, 64):  # number of batches per epoch
            model, opt_state, train_loss = train_step(model, loss_fn, optim, opt_state, batch)

        epoch_stats = f"Epoch {epoch:3d} | Train Loss: {train_loss:.5f}"

        if test_data:
            test_loss = jnp.sqrt(loss_fn(model, test_data))
            epoch_stats += f" | Test Loss: {test_loss:.5f}"

        pbar.set_description(epoch_stats)


    return model
