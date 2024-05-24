from typing import Optional

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from diff_ml.typing import Model


class Normalization(eqx.Module):
    """Preprocessing layer to receive normalized input.

    Similar to `Normalization` layer of Keras.
    """

    mean: Float[Array, ""]
    std: Float[Array, ""]

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """Normalizes (input) data.

        Args:
            x: Input data.
            key: Random number generator key (not used).

        Returns:
            Normalized input.
        """
        del key
        return (x - self.mean) / self.std


class Denormalization(eqx.Module):
    """Preprocessing layer to denormalize data to original scale."""

    mean: Float[Array, ""]
    std: Float[Array, ""]

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """Denormalizes (output) data.

        Args:
            x: Output data.
            key: Random number generator key (not used).

        Returns:
            Denormalized output.
        """
        del key
        return x * self.std + self.mean


class Normalized(eqx.Module):
    """Preprocessing layer wrapping a model with normalization and denormalization layers."""

    seq: eqx.nn.Sequential

    def __init__(
        self,
        x_normalizer: Normalization,
        model: Model,
        y_denormalizer: Denormalization,
    ):
        """Initialization.

        Args:
            x_normalizer: Normalization layer.
            model: Model.
            y_denormalizer: Denormalization layer.
        """
        self.seq = eqx.nn.Sequential(layers=(x_normalizer, model, y_denormalizer))

    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """Forward pass."""
        return self.seq(x, key=key)
