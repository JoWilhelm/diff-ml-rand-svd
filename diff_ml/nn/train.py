import equinox as eqx
import jax.numpy as jnp
import optax
from jax import vmap
from jaxtyping import Array, Float, Int, PyTree

from diff_ml import Data, DataGenerator
from diff_ml.nn.loss import mse
from diff_ml.nn.modules import MakeScalar


@eqx.filter_jit
def loss_fn(model: eqx.Module, x: Float[Array, " batch"], y: Float[Array, " batch"]) -> Float:
    pred_y = vmap(MakeScalar(model))(x)
    pred_y = pred_y[:, jnp.newaxis]
    result = mse(y, pred_y)
    return result


def train(
    model: eqx.Module,
    train_ds: DataGenerator,
    optim: optax.GradientTransformation,
    n_epochs: Int,
    # test_ds: DataGenerator
) -> PyTree:
    @eqx.filter_jit
    def train_step(model: PyTree, opt_state: PyTree, batch: Data):
        x, y = batch.values()
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    for epoch in range(n_epochs):
        for batch in train_ds:
            model, opt_state, train_loss = train_step(model, opt_state, batch)

            # test_loss = jnp.sqrt(evaluate(model, generator_test_ds))
            print(
                f"Finished epoch {epoch:3d}",
                f" | Train Loss: {train_loss:.5f}",
                # ' | Test Loss: {:.5f}'.format(test_loss),
            )

    return model
