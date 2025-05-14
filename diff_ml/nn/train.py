from itertools import islice
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
from jax import vmap
from jaxtyping import Float, Array, ArrayLike, PyTree
from tqdm import tqdm

from diff_ml.typing import Data, DataGenerator




# TODO put this somewhere where it makes more sense
class LossState(eqx.Module):
    losses: Float[Array, "n_losses"] = jnp.ones(3) * 1/3
    lambdas: Float[Array, "n_losses"] = jnp.ones(3) * 1/3
    initial_losses: Float[Array, "n_losses"] = jnp.ones(3) * 1/3
    accum_losses: Float[Array, "n_losses"] = jnp.zeros(3)
    prev_mean_losses: Float[Array, "n_losses"] = jnp.zeros(3)
    current_iter: Float[Array, "n_losses"] = jnp.zeros(3)

    def update_losses(state, new_value: Float[Array, "n_losses"]): # -> LossState:
        return LossState(new_value, state.lambdas, state.initial_losses, state.accum_losses, state.prev_mean_losses, state.current_iter)

    def update_lambdas(state, new_value: Float[Array, "n_losses"]):# -> LossState:
        return LossState(state.losses, new_value, state.initial_losses, state.accum_losses, state.prev_mean_losses,  state.current_iter)
        
    def update_initial_losses(state, new_value: Float[Array, "n_losses"]): # -> LossState:
        return LossState(state.losses, state.lambdas, new_value, state.accum_losses, state.prev_mean_losses,  state.current_iter)
        
    def update_accum_losses(state, new_value: Float[Array, "n_losses"]): # -> LossState:
        return LossState(state.losses, state.lambdas, state.initial_losses, new_value, state.prev_mean_losses,  state.current_iter)
        
    def update_current_iter(state, new_value: Float[Array, "n_losses"]): # -> LossState:
        return LossState(state.losses, state.lambdas, state.initial_losses, state.accum_losses, state.prev_mean_losses,  new_value)

    def update_prev_mean_losses(state, new_value: Float[Array, "n_losses"]): # -> LossState:
        return LossState(state.losses, state.lambdas, state.initial_losses, state.accum_losses, new_value,  state.current_iter)

    def __repr__(self):
        return "LossState"




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
def train_step(model, loss_fn, optim: optax.GradientTransformation, opt_state: PyTree, batch: Data, loss_state: LossState):
    """Canonical training step for a single batch."""
    (loss_value, loss_state), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, batch, loss_state)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_value, loss_state


def metrics_update_element(metrics: dict, key: str, epoch: int, loss: ArrayLike):
    metrics[key] = metrics[key].at[epoch].set(loss)


def train(
    model,
    loss_fn,
    train_data: DataGenerator,
    eval_fn,
    test_data: Optional[Data],
    optim: optax.GradientTransformation,
    n_epochs: int,
    n_batches_per_epoch: int = 64,
) -> PyTree:
    """Canonical training loop."""
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    train_loss = jnp.zeros(1)
    batch_size = len(next(train_data)["x"])
    metrics = {"train_loss": jnp.zeros(n_epochs), "test_loss": jnp.zeros(n_epochs)}

    loss_state = LossState()
    loss_state = LossState(jnp.array([0.0, 0.0, 1.0]), jnp.array([1/3, 1/3, 1/3]), jnp.array([0.0, 0.0, 1.0]), jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0])) 


    pbar = tqdm(range(n_epochs))
    for epoch in pbar:
        for batch in islice(train_data, n_batches_per_epoch):
            model, opt_state, train_loss, loss_state = train_step(model, loss_fn, optim, opt_state, batch, loss_state)


        # update loss_state
        loss_state = loss_state.update_prev_mean_losses(loss_state.accum_losses / loss_state.current_iter[0])
        loss_state = loss_state.update_accum_losses(jnp.zeros(len(loss_state.losses)))
        loss_state = loss_state.update_current_iter(jnp.zeros(len(loss_state.losses)))


        metrics_update_element(metrics, "train_loss", epoch, train_loss)
        epoch_stats = f"Epoch: {epoch:3d} | Train: {train_loss:.5f}"

        if test_data:
            test_loss = evaluate(model, test_data, batch_size, eval_fn)
            metrics_update_element(metrics, "test_loss", epoch, test_loss)
            epoch_stats += f" | Test: {test_loss:.5f}"

        pbar.set_description(epoch_stats)

    return model, metrics
