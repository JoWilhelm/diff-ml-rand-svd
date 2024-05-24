from itertools import islice
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jax import vmap
from jaxtyping import PyTree
from tqdm import tqdm

from diff_ml.typing import Data, DataGenerator


@eqx.filter_jit
def evaluate(model, test_data, n_batch_size, loss_fn):
    """Compute the average loss of the model on the test data."""
    n_elements = test_data["x"].shape[0]
    n = n_elements // n_batch_size

    batched_test_data = jtu.tree_map(lambda x: jnp.reshape(x, (n, n_batch_size, *x.shape[1:])), test_data)
    flattened_data, treedef = jtu.tree_flatten(batched_test_data)

    def batch_eval(data):
        batch = jtu.tree_unflatten(treedef, data)
        return loss_fn(model, batch)

    losses = vmap(batch_eval)(flattened_data)
    avg_loss = jax.lax.associative_scan(jnp.add, losses)[-1] / n
    return avg_loss


@eqx.filter_jit
def train_step(model, loss_fn, optim: optax.GradientTransformation, opt_state: PyTree, batch: Data):
    """Canonical training step for a single batch."""
    loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value


def train(
    model,
    loss_fn,
    train_data: DataGenerator,
    eval_fn,
    test_data: Optional[Data],
    optim: optax.GradientTransformation,
    n_epochs: int,
) -> PyTree:
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    train_loss = jnp.zeros(1)
    batch_size = len(next(train_data)["x"])

    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        for batch in islice(train_data, 64):  # number of batches per epoch
            model, opt_state, train_loss = train_step(model, loss_fn, optim, opt_state, batch)

        epoch_stats = f"Epoch: {epoch:3d} | Train: {train_loss:.5f}"

        if test_data:
            test_loss = evaluate(model, test_data, batch_size, eval_fn)
            epoch_stats += f" | Test: {test_loss:.5f}"

        pbar.set_description(epoch_stats)

    return model
