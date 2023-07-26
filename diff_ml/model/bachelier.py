from dataclasses import dataclass

import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
import numpy as np
from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray

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
    vol_mult: float = 1.5
    vol_bkt: float = 0.2
    anti: bool = False


@dataclass
class DifferentialData:
    """Differential data."""

    xs: Float[ArrayLike, " n"]
    ys: Float[ArrayLike, " n"]
    zs: Float[ArrayLike, " n"]


def generate_correlated_samples(key: PRNGKeyArray, n_samples: int) -> Array:
    """TODO: ."""
    data = jrandom.uniform(key, shape=(2 * n_samples, n_samples), minval=-1.0, maxval=1.0)
    covariance = data.T @ data
    inv_vols = jnp.diag(1.0 / jnp.sqrt(jnp.diagonal(covariance)))
    return jnp.linalg.multi_dot([inv_vols, covariance, inv_vols])


@dataclass
class Bachelier:
    """Bachelier model.

    Reference: https://en.wikipedia.org/wiki/Bachelier_model

    TODO: add iwasawa.us/normal.pdf
    """

    key: PRNGKeyArray
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

    class AnalyticalCall:
        """Analytical solution of Bachelier."""

        @staticmethod
        def price(spot, strike, vol, t):
            r"""Analytical solution to the undiscounted call option price.

            As in equation (3) of https://arxiv.org/pdf/2104.08686.pdf.

            Args:
                spot: the spot price, also denoted as $S_0$.
                strike: an array of strike prices, also denoted as $K$.
                vol: volatility, also denoted as $\sigma_N$.
                t: time to maturity, also denoted as $T - t$ or $T$.


            Returns:
                TODO
            """
            sqrt_t = jnp.sqrt(t)
            d = (spot - strike) / (vol * sqrt_t)
            normal_cdf_d = jstats.norm.cdf(d)
            normal_pdf_d = jstats.norm.pdf(d)
            price = vol * sqrt_t * (d * normal_cdf_d + normal_pdf_d)
            return price

        @staticmethod
        def delta(spot, strike, vol, t):
            r"""Analytical delta.

            The delta is the derivative of the price sensitivity w.r.t. the spot price.

            As in 5.1 of https://arxiv.org/pdf/2104.08686.pdf.

            Args:
                spot: the spot price, also denoted as $S_0$.
                strike: an array of strike prices, also denoted as $K$.
                vol: volatility, also denoted as $\sigma_N$.
                t: time to maturity, also denoted as $T - t$ or $T$.


            Returns:
                TODO
            """
            d = (spot - strike) / (vol * jnp.sqrt(t))
            return jstats.norm.cdf(d)

        @staticmethod
        def gamma(spot, strike, vol, t):
            r"""Analytical gamma.

            The gamma is the 2nd-order derivative of the price
            sensitivity w.r.t. the spot price.

            As in 5.1 of https://arxiv.org/pdf/2104.08686.pdf.

            Args:
                spot: the spot price, also denoted as $S_0$.
                strike: an array of strike prices, also denoted as $K$.
                vol: volatility, also denoted as $\sigma_N$.
                t: time to maturity, also denoted as $T - t$ or $T$.


            Returns:
                TODO
            """
            d = (spot - strike) / (vol * jnp.sqrt(t))
            return jstats.norm.pdf(d) / (vol * jnp.sqrt(t))

        @staticmethod
        def vega(spot, strike, vol, t):
            r"""Analytical vega.

            The vega is the 2nd-order derivative of the price
            sensitivity w.r.t. the volatility.

            As in 5.1 of https://arxiv.org/pdf/2104.08686.pdf.

            Args:
                spot: an array of spot prices, also denoted as $S_0$.
                strike: an array of strike prices, also denoted as $K$.
                vol: volatility, also denoted as $\sigma_N$.
                t: time to maturity, also denoted as $T - t$ or $T$.


            Returns:
                TODO
            """
            d = (spot - strike) / (vol * jnp.sqrt(t))
            return jnp.sqrt(t) * jstats.norm.pdf(d)


def main():
    dml.print_df()
    return dml.mol()


if __name__ == "__main__":
    main()
