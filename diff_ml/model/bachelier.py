from functools import partial
from typing import Final

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats
from jaxtyping import Array, Float, PRNGKeyArray, ScalarLike

import diff_ml as dml
from diff_ml import Data, DataGenerator
from diff_ml.model.payoff import EuropeanPayoff


def generate_correlation_matrix(key: PRNGKeyArray, n_samples: int) -> Array:
    """TODO: ."""
    data = jrandom.uniform(key, shape=(2 * n_samples, n_samples), minval=-1.0, maxval=1.0)
    covariance = data.T @ data
    inv_vols = jnp.diag(1.0 / jnp.sqrt(jnp.diagonal(covariance)))
    return jnp.linalg.multi_dot([inv_vols, covariance, inv_vols])


# TODO: seperate the analytic part into a seperate class
# @dataclass
class Bachelier:
    """Bachelier model.

    References:
        https://en.wikipedia.org/wiki/Bachelier_model

        https://iwasawa.us/normal.pdf

    Attributes:
        key: a key for the random number generator of jax.
        n_dims: number of dimensions. A dimension usually corresponds to an asset price.
        weights: an array of weights indicating the importance
            of each dimension of the spots.
        t_exposure: the start time you get exposed to the option.
        t_maturity: the time the option will expire, i.e. reach its maturity.
        strike_price: the strike price, often refered to as $K$.
        vol_mult: the volatility multiplier. If above 1, more data will be generated on the wings.
        vol_basket: the volatility of the basket. Used to normalize the volatilities.
    """

    key: PRNGKeyArray
    n_dims: Final[int]
    weights: Float[Array, " n_dims"]

    t_exposure: float = 1.0
    t_maturity: float = 2.0
    strike_price: float = 1.10
    vol_mult: float = 1.5
    vol_basket: float = 0.2
    use_antithetic: bool = True

    def __init__(self, key, n_dims, weights):
        """TODO: ."""
        self.key = key
        self.n_dims = n_dims

        # scale weights to sum up to 1
        self.weights = weights / jnp.sum(weights)

        self.use_antithetic = False

    def baskets(self, spots):
        """TODO: ."""
        return jnp.dot(spots, self.weights).reshape((-1, 1))

    @staticmethod
    def path_simulation():
        """TODO: ."""
        pass

    @staticmethod
    def payoff_analytic_differentials(xs, paths, weights, strike_price):
        """TODO: ."""
        spots_end = xs + paths
        baskets_end = jnp.dot(spots_end, weights)
        analytic_differentials = jnp.where(baskets_end > strike_price, 1.0, 0.0)
        analytic_differentials = analytic_differentials.reshape((-1, 1))
        weights = weights.reshape((1, -1))
        # TODO: Replace either with jnp.multiply or jnp.matmul.
        #       Make sure to use the correct one! Here it doesn't
        #       matter since we have (x, 1) but this makes it clearer
        #       what the intention behind this operation is.
        result = analytic_differentials * weights
        return result

    @staticmethod
    def payoff_antithetic_analytic_differentials(xs, paths, weights, strike_price):
        """TODO: ."""
        spots_end_a = xs + paths
        baskets_end_a = jnp.dot(spots_end_a, weights)
        spots_end_b = xs - paths
        baskets_end_b = jnp.dot(spots_end_b, weights)

        differentials_a = jnp.where(baskets_end_a > strike_price, 1.0, 0.0).reshape((-1, 1)) * weights.reshape((1, -1))
        differentials_b = jnp.where(baskets_end_b > strike_price, 1.0, 0.0).reshape((-1, 1)) * weights.reshape((1, -1))
        differentials = 0.5 * (differentials_a + differentials_b)
        return differentials

    @staticmethod
    def payoff(
        xs: Float[Array, "n_samples n_dims"],
        paths: Float[Array, "n_samples n_dims"],
        weights: Float[Array, " n_dims"],
        strike_price: Float[ScalarLike, ""],
    ) -> Float[Array, " n_samples"]:
        """TODO: ."""
        spots_end = xs + paths
        baskets_end = jnp.dot(spots_end, weights)
        pay = EuropeanPayoff.call(baskets_end, strike_price)
        return pay

    @staticmethod
    def antithetic_payoff(
        xs: Float[Array, "n_samples n_dims"],
        paths: Float[Array, "n_samples n_dims"],
        weights: Float[Array, " n_dims"],
        strike_price: Float[ScalarLike, ""],
    ) -> Float[Array, " n_samples"]:
        """TODO: ."""
        spots_end_a = xs + paths
        baskets_end_a = jnp.dot(spots_end_a, weights)
        pay_a = EuropeanPayoff.call(baskets_end_a, strike_price)

        spots_end_b = xs - paths
        baskets_end_b = jnp.dot(spots_end_b, weights)
        pay_b = EuropeanPayoff.call(baskets_end_b, strike_price)

        pay = 0.5 * (pay_a + pay_b)
        return pay

    def sample(self, n_samples: int) -> Data:
        """TODO: ."""
        self.key, subkey = jrandom.split(self.key)
        #  w.l.o.g., initialize spots, i.e. S_0, as all ones
        spots_0 = jnp.repeat(1.0, self.n_dims)

        # generate random correlation matrix
        correlated_samples = generate_correlation_matrix(subkey, self.n_dims)

        # TODO: consider using cupy for random number generation in MC simulation
        #       in general we should extract the random number generator to be agnostic

        # generate random volatilities
        self.key, subkey = jrandom.split(self.key)
        vols = jrandom.uniform(subkey, shape=(self.n_dims,), minval=5.0, maxval=50.0)

        # W.l.o.g., normalize the volatilities for a given volatility of the basket.
        # It makes plotting the data more convenient.
        normalized_vols = (self.weights * vols).reshape((-1, 1))
        v = jnp.sqrt(jnp.linalg.multi_dot([normalized_vols.T, correlated_samples, normalized_vols]).reshape(1))
        vols = vols * self.vol_basket / v

        t_delta = self.t_maturity - self.t_exposure

        # Choleski
        diag_v = jnp.diag(vols)
        cov = jnp.linalg.multi_dot([diag_v, correlated_samples, diag_v])
        chol = jnp.linalg.cholesky(cov) * jnp.sqrt(t_delta)

        # increase vols for simulation of xs so we have more samples in the wings
        chol_0 = chol * self.vol_mult * jnp.sqrt(self.t_exposure / t_delta)

        # simulations
        self.key, subkey = jrandom.split(self.key)
        normal_samples = jrandom.normal(subkey, shape=(2, n_samples, self.n_dims))
        paths_0 = normal_samples[0] @ chol_0.T
        paths_1 = normal_samples[1] @ chol.T
        spots_1 = spots_0 + paths_0

        if self.use_antithetic:
            analytic_differentials_fn = Bachelier.payoff_antithetic_analytic_differentials
            payoff_fn = Bachelier.antithetic_payoff
        else:
            analytic_differentials_fn = Bachelier.payoff_analytic_differentials
            payoff_fn = Bachelier.payoff

        differentials_analytic = analytic_differentials_fn(spots_1, paths_1, self.weights, self.strike_price)
        payoff_fn = partial(payoff_fn, weights=self.weights, strike_price=self.strike_price)

        payoffs_vjp, vjp_fn = jax.vjp(payoff_fn, spots_1, paths_1)
        differentials_vjp = vjp_fn(jnp.ones(payoffs_vjp.shape))[0]

        assert jnp.allclose(differentials_analytic, differentials_vjp)  # noqa: S101

        return {"spot": spots_1, "payoff": payoffs_vjp, "differentials": differentials_vjp}

    def dataloader(self):
        """Yields from already computed data."""
        # TODO: Implement
        pass

    def batch_generator(self, n_batch: int):
        """Generates a batch of data on the fly."""
        while True:
            yield self.sample(n_batch)

    def generator(self, n_precompute: int) -> DataGenerator:
        """Generates new data on the fly.

        Note that this generator continues forever. The `n_precompute` parameter is only
        used to control the number of samples that are computed at once. The generator
        will then yield `n_precompute` times before computing the next set of data points.

        Args:
            n_precompute: number of samples to generate at once.

        Yields:
            A Data object.
        """
        while True:
            samples = self.sample(n_precompute)
            keys = samples.keys()
            values = samples.values()

            for i in range(n_precompute):
                ith_sample = (v[i] for v in values)
                sample = dict(zip(keys, ith_sample))
                yield sample

    def analytic(self, n_samples, minval=0.5, maxval=1.5) -> Data:
        """TODO: ."""
        # adjust lower and upper for dimension
        adj = 1 + 0.5 * jnp.sqrt((self.n_dims - 1) * (maxval - minval) / 12)
        adj_lower = 1.0 - (1.0 - minval) * adj
        adj_upper = 1.0 + (maxval - 1.0) * adj

        # draw random spots within range
        self.key, subkey = jrandom.split(self.key)
        spots = jrandom.uniform(subkey, shape=(n_samples, self.n_dims), minval=adj_lower, maxval=adj_upper)
        baskets = jnp.dot(spots, self.weights).reshape((-1, 1))
        time_to_maturity = self.t_maturity - self.t_exposure
        prices = Bachelier.Call.price(baskets, self.strike_price, self.vol_basket, time_to_maturity)
        prices = prices.reshape((-1,))
        # prices = prices.reshape((-1, 1))

        # in analytical solution we directly compute greeks w.r.t. the basket price
        greeks = Bachelier.Call.greeks(baskets, self.strike_price, self.vol_basket, time_to_maturity)
        return {"spot": spots, "payoff": prices, "differentials": greeks[0].reshape((-1,))}

    class Call:
        """Analytic solutions to price and greeks (delta, gamma, vega) of call option on Bachelier."""

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
        def delta(spot, strike, vol, t) -> Array:
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
        def gamma(spot, strike, vol, t) -> Array:
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
        def vega(spot, strike, vol, t) -> Array:
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

        @staticmethod
        def greeks(spot, strike, vol, t) -> tuple[Array, Array, Array]:
            r"""Greeks.

            As in 5.1 of https://arxiv.org/pdf/2104.08686.pdf.

            Args:
                spot: an array of spot prices, also denoted as $S_0$.
                strike: an array of strike prices, also denoted as $K$.
                vol: volatility, also denoted as $\sigma_N$.
                t: time to maturity, also denoted as $T - t$ or $T$.


            Returns:
                TODO
            """
            call = Bachelier.Call
            deltas = call.delta(spot, strike, vol, t)
            gammas = call.gamma(spot, strike, vol, t)
            vegas = call.vega(spot, strike, vol, t)
            return deltas, gammas, vegas


def main():
    dml.print_df()
    return dml.mol()


if __name__ == "__main__":
    main()
