from typing import Protocol

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float


class Payoff(Protocol):
    """TODO: ."""

    @staticmethod
    def call(maturity_prices: Float[ArrayLike, " n"], strike_price: float) -> Float[ArrayLike, " n"]:
        """TODO: ."""
        ...

    @staticmethod
    def put(maturity_prices: Float[ArrayLike, " n"], strike_price: float) -> Float[ArrayLike, " n"]:
        """TODO: ."""
        ...


class EuropeanPayoff:
    """TODO: ."""

    @staticmethod
    def call(maturity_prices: Float[ArrayLike, " n"], strike_prices: float) -> Float[Array, " n"]:
        """TODO: ."""
        return jnp.maximum(maturity_prices - strike_prices, 0.0)

    @staticmethod
    def put(maturity_prices: Float[ArrayLike, " n"], strike_prices: float) -> Float[Array, " n"]:
        """TODO: ."""
        return jnp.maximum(strike_prices - maturity_prices, 0.0)
