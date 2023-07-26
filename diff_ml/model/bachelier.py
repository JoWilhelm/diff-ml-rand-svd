from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
import numpy as np
from jaxtyping import ArrayLike, Float, Key

import diff_ml as dml


@dataclass(eq=True, frozen=True)
class BachelierParams:
    """Bachelier model parameters."""

    n_dim: int = 1
    t_exposure: float = 1.0
    t_maturity: float = 2.0
    strike_price: float = 1.10
    test_set_lb: float = 0.5
    test_set_ub: float = 1.5
    vola_mult: float = 1.5
    vola_bkt: float = 0.2
    anti: bool = False


@dataclass
class DifferentialData:
    """Differential data."""

    xs: Float[ArrayLike, " n"]
    ys: Float[ArrayLike, " n"]
    zs: Float[ArrayLike, " n"]


@dataclass
class Bachelier:
    """Bachelier model.

    Reference: https://en.wikipedia.org/wiki/Bachelier_model

    TODO: add iwasawa.us/normal.pdf
    """

    key: Key
    params: BachelierParams

    def payoff(self):
        """TODO: ."""
        pass

    def sample(self, n_samples: int) -> DifferentialData:
        """TODO: ."""
        self.key, subkey = jrandom.split(self.key)
        return DifferentialData(np.zeros(n_samples), np.zeros(n_samples), np.zeros(n_samples))

    def generator(self):
        """TODO: ."""
        pass

    class Analytical:
        """Analytical solution of Bachelier."""

        @staticmethod
        def price(spot, strike, vol, t):
            """Analytical price."""
            d = (spot - strike) / vol / jnp.sqrt(t)
            return vol * jnp.sqrt(t) * (d * jstats.norm.cdf(d) + jstats.norm.pdf(d))

        @staticmethod
        def delta(spot, strike, vol, t):
            """Analytical delta."""
            d = (spot - strike) / vol / jnp.sqrt(t)
            return jstats.norm.cdf(d)

        @staticmethod
        def vega(spot, strike, vol, t):
            """Analytical vega."""
            d = (spot - strike) / vol / jnp.sqrt(t)
            return jnp.sqrt(t) * jstats.norm.pdf(d)

        @staticmethod
        def gamma(spot, strike, vol, t):
            """Analytical gamma."""
            d = (spot - strike) / vol / jnp.sqrt(t)
            return jstats.norm.pdf(d) / (vol * jnp.sqrt(t))


def main():
    dml.print_df()
    return dml.mol()


if __name__ == "__main__":
    main()
