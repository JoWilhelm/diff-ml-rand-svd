from dataclasses import dataclass
from typing import Callable, Literal, Optional, Protocol, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jax import jit
from jaxtyping import Array, PRNGKeyArray


Model = Callable[[Array, Optional[PRNGKeyArray]], Array]


class MakeScalar(eqx.Module):
    """Turn the output of a model to a scalar.

    It allows for a model to be used in jax.grad and similar
    functions of jax requiring a function with a scalar output.
    This only works if the output of the model is an `Array` with
    a single element.
    """

    model: Model

    def __call__(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        if out.shape[-1] != 1:
            msg = "The model must return an array with a single element for MakeScalar to be applicable."
            raise ValueError(msg)

        return jnp.reshape(out, ())


class Normalization(eqx.Module):
    """Preprocessing layer to receive normalized input.

    Similar to `Normalization` layer of Keras.
    """

    mean: float
    std: float

    def __call__(self, x: Array) -> Array:
        return (x - self.mean) / self.std


class Denormalization(eqx.Module):
    """Preprocessing layer to denormalize data to original scale."""

    mean: float
    std: float

    def __call__(self, x: Array) -> Array:
        return x * self.std + self.mean


class Normalized(eqx.Module):
    seq: eqx.nn.Sequential

    def __init__(self, x_normalizer: Normalization, model: eqx.nn.MLP, y_denormalizer: Denormalization):
        self.seq = eqx.nn.Sequential(x_normalizer, model, y_denormalizer)

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        return self.seq(x, key=key)
