import jax.numpy as jnp
from jaxtyping import Array, Float


class EuropeanPayoff:
    """TODO: ."""

    @staticmethod
    def call(maturity_prices: Float[Array, " n"], strike_prices: float):
        """TODO: ."""
        return jnp.maximum(maturity_prices - strike_prices, 0.0)

    @staticmethod
    def put(maturity_prices: Float[Array, " n"], strike_prices: float):
        """TODO: ."""
        return jnp.maximum(strike_prices - maturity_prices, 0.0)
