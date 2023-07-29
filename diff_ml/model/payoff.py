from typing import Protocol

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, ScalarLike


class Payoff(Protocol):
    """TODO: ."""

    @staticmethod
    def call(maturity_prices: Float[ArrayLike, " n"], strike_price: Float[ScalarLike, ""]) -> Float[ArrayLike, " n"]:
        """TODO: ."""
        ...

    @staticmethod
    def put(maturity_prices: Float[ArrayLike, " n"], strike_price: Float[ScalarLike, ""]) -> Float[ArrayLike, " n"]:
        """TODO: ."""
        ...


class EuropeanPayoff:
    """TODO: ."""

    @staticmethod
    def call(maturity_prices: Float[ArrayLike, " n"], strike_prices: Float[ScalarLike, ""]) -> Float[Array, " n"]:
        """TODO: ."""
        return jnp.maximum(jnp.subtract(maturity_prices, strike_prices), 0.0)

    @staticmethod
    def put(maturity_prices: Float[ArrayLike, " n"], strike_prices: Float[ScalarLike, ""]) -> Float[Array, " n"]:
        """TODO: ."""
        return jnp.maximum(jnp.subtract(strike_prices, maturity_prices), 0.0)
