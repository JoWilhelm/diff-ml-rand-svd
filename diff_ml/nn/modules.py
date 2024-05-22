from collections.abc import Callable
from typing import Optional

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


Model = Callable[[Array, Optional[PRNGKeyArray]], Array]


class Normalization(eqx.Module):
    """Preprocessing layer to receive normalized input.

    Similar to `Normalization` layer of Keras.
    """

    mean: Float[Array, ""]
    std: Float[Array, ""]

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        del key
        return (x - self.mean) / self.std


class Denormalization(eqx.Module):
    """Preprocessing layer to denormalize data to original scale."""

    mean: Float[Array, ""]
    std: Float[Array, ""]

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        del key
        return x * self.std + self.mean


class Normalized(eqx.Module):
    seq: eqx.nn.Sequential

    def __init__(
        self,
        x_normalizer: Normalization,
        model: eqx.Module,
        y_denormalizer: Denormalization,
    ):
        self.seq = eqx.nn.Sequential(layers=(x_normalizer, model, y_denormalizer))

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        return self.seq(x, key=key)
