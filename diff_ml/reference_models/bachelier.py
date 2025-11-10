import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.scipy.stats as jstats

from jaxtyping import Array, ArrayLike, Float, PRNGKeyArray, ScalarLike

from diff_ml.reference_models.reference_model_class import ReferenceModel
from diff_ml.utils import rmse
from diff_ml.typing import DifferentialData, Scalar

from functools import partial
import matplotlib.pyplot as plt



"""
TODO

credit: Neil Kichler
"""


class EuropeanPayoff:
    @staticmethod
    def call(maturity_prices: Float[ArrayLike, " n"], strike_prices: Float[ScalarLike, ""]) -> Float[Array, " n"]:
        return jnp.maximum(jnp.subtract(maturity_prices, strike_prices), 0.0)
    @staticmethod
    def put(maturity_prices: Float[ArrayLike, " n"], strike_prices: Float[ScalarLike, ""]) -> Float[Array, " n"]:
        return jnp.maximum(jnp.subtract(strike_prices, maturity_prices), 0.0)
    @staticmethod
    def smoothed_call(
        maturity_prices: Float[ArrayLike, " n"],
        strike_prices: Float[ScalarLike, ""],
        eps = 0.01,
    ) -> Float[Array, " n"]:
        """
        C^2-smoothed call payoff.

        - For S <= K - eps:        0
        - For S >= K + eps:        S - K
        - For |S - K| < eps:       smooth polynomial bridge
        """
        S = jnp.asarray(maturity_prices)
        K = jnp.asarray(strike_prices)
        eps = jnp.asarray(eps)

        # distance to strike
        t = S - K

        # polynomial segment on (-eps, eps)
        # p(t) = -t^4/(16 eps^3) + 3 t^2/(8 eps) + (1/2) t + 3 eps/16
        inner = (
            - (t ** 4) / (16.0 * eps ** 3)
            + 3.0 * (t ** 2) / (8.0 * eps)
            + 0.5 * t
            + 3.0 * eps / 16.0
        )

        zero = jnp.zeros_like(t)
        linear = t  # == (S - K)

        return jnp.where(
            t <= -eps,
            zero,
            jnp.where(t >= eps, linear, inner),
        )

def generate_correlation_matrix(key: PRNGKeyArray, n_samples: int) -> Array:
    """TODO: ."""
    data = jrandom.uniform(key, shape=(2 * n_samples, n_samples), minval=-1.0, maxval=1.0)
    covariance = data.T @ data
    inv_vols = jnp.diag(1.0 / jnp.sqrt(jnp.diagonal(covariance)))
    return jnp.linalg.multi_dot([inv_vols, covariance, inv_vols])



class Bachelier(ReferenceModel):
    """Bachelier model.

    References:
        https://en.wikipedia.org/wiki/Bachelier_model

        https://iwasawa.us/normal.pdf

    Attributes:
        key: a key for the random number generator of jax.
        n_dims: number of dimensions. A dimension usually corresponds to an asset price.
        weights: an array of weights indicating the importance
            of each dimension of the spots, i.e. the weight of the asset in the basket.
        t_exposure: the start time you get exposed to the option.
        t_maturity: the time the option will expire, i.e. reach its maturity.
        strike_price: the strike price, often refered to as $K$.
        vol_mult: the volatility multiplier. If above 1, more data will be generated on the wings.
        vol_basket: the volatility of the basket. Used to normalize the volatilities.
    """

    key_test: PRNGKeyArray
    key_train: PRNGKeyArray
    n_dims: int
    basket_dim: int
    weights: Float[Array, "basket_dim"]

    t_exposure: float = 1.0
    t_maturity: float = 2.0
    strike_price: float = 1.10
    vol_mult: float = 1.5
    vol_basket: float = 0.2
    use_antithetic: bool = True
    was_normalized: bool = False

    def __init__(self, key, basket_dim, weights):
        """TODO: ."""
        if basket_dim != len(weights):
            val = f"Mismatch in number of dimensions ({basket_dim}) and number of weights ({weights}) given."
            raise ValueError(val)

        self.key_test, self.key_train = jax.random.split(key, 2)
        self.basket_dim = basket_dim
        self.un_flattened_shape = [basket_dim]

        # scale weights to sum up to 1
        self.weights = weights / jnp.sum(weights)
        self.n_dims = basket_dim


        # fix cov once
        key, subkey = jrandom.split(self.key_train)
        correlated_samples = generate_correlation_matrix(subkey, self.n_dims)
        # generate random volatilities
        key, subkey = jrandom.split(key)
        vols = jrandom.uniform(subkey, shape=(self.n_dims,), minval=5.0, maxval=50.0)
        # W.l.o.g., normalize the volatilities for a given volatility of the basket.
        # It makes plotting the data more convenient.
        normalized_vols = (self.weights * vols).reshape((-1, 1))
        v = jnp.sqrt(jnp.linalg.multi_dot([normalized_vols.T, correlated_samples, normalized_vols]).reshape(1))
        vols = vols * self.vol_basket / v
        diag_v = jnp.diag(vols)
        self.cov = jnp.linalg.multi_dot([diag_v, correlated_samples, diag_v])
        
   
   
   
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
        pay = EuropeanPayoff.smoothed_call(baskets_end, strike_price)
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
        pay_a = EuropeanPayoff.smoothed_call(baskets_end_a, strike_price)

        spots_end_b = xs - paths
        baskets_end_b = jnp.dot(spots_end_b, weights)
        pay_b = EuropeanPayoff.smoothed_call(baskets_end_b, strike_price)

        pay = 0.5 * (pay_a + pay_b)
        return pay
    


    def analytic_basket_price_single_x(self, x) -> Scalar:
        basket = jnp.dot(x, self.weights).reshape((-1, 1))
        time_to_maturity = self.t_maturity - self.t_exposure
        price = Bachelier.Call.price(
                        spot =basket,
                       strike=self.strike_price,
                       vol=self.vol_basket,
                       t=time_to_maturity
                )
        price = price.reshape((-1,))
        return price[0]
        
        

    def simulated_basket_price_single_x(self, x) -> Scalar:
        n_paths = 1000
        
        x = jnp.asarray(x)
        cov = self.cov

        t_delta = self.t_maturity - self.t_exposure

        # simulations using fixed cov and seed for paths
        chol = jnp.linalg.cholesky(cov) * jnp.sqrt(t_delta)
        normal_samples = jrandom.normal(self.key_train, shape=(n_paths, self.n_dims))
        paths = normal_samples @ chol.T
        
        
        if self.use_antithetic:
            payoff_fn = Bachelier.antithetic_payoff
        else:
            payoff_fn = Bachelier.payoff

        payoff_fn = partial(payoff_fn, weights=self.weights, strike_price=self.strike_price)
        
        payoffs = payoff_fn(x[jnp.newaxis, :], paths)

        return jnp.mean(payoffs, axis=0)





    def reference_fn(self):
        #return self.analytic_basket_price_single_x 
        return partial(self.simulated_basket_price_single_x)



    def sample(self, key:PRNGKeyArray, n_samples:int, order=1) -> DifferentialData:
        if order > 1:
            raise ValueError("Differential data of order > 1 not supported via sample(). Use analytic() for that, e.g. for test set generation.")

        cov = self.cov
        spots_0 = jnp.repeat(1.0, self.n_dims)
        t_delta = self.t_maturity - self.t_exposure
        # simulations using fixed cov and seed for paths
        chol = jnp.linalg.cholesky(cov) * jnp.sqrt(t_delta)
        # increase vols for simulation of xs so we have more samples in the wings
        chol_0 = chol * self.vol_mult * jnp.sqrt(self.t_exposure / t_delta)
        # fresh batch key for S0, fixed key for paths
        normals_x = jrandom.normal(key, shape=(n_samples, self.n_dims))
        paths_0 = normals_x @ chol_0.T
        x = spots_0 + paths_0
        

        value_and_grad_fn = jax.value_and_grad(self.reference_fn())
        y, dy = jax.vmap(value_and_grad_fn)(x)


        return DifferentialData(
            order = 1,
            x = x,
            y = y,
            dy = dy
        )






#    def sample(self, key: PRNGKeyArray, n_samples: int, order=1) -> DifferentialData:
#        """TODO: ."""
#
#        if order > 1:
#            raise ValueError("Differential data of order > 1 not supported via sample(). Use analytic() for that, e.g. for test set generation.")
#
#        n_paths = 100
#        
#        cov = self.cov
#        
#        
#        
#        #  w.l.o.g., initialize spots, i.e. S_0, as all ones
#        spots_0 = jnp.repeat(1.0, self.n_dims)
#        t_delta = self.t_maturity - self.t_exposure
#        
#        
#        
#        ## generate random correlation matrix
#        #key, subkey = jrandom.split(key)
#        #correlated_samples = generate_correlation_matrix(subkey, self.n_dims)
#        #
#        ## generate random volatilities
#        #key, subkey = jrandom.split(key)
#        #vols = jrandom.uniform(subkey, shape=(self.n_dims,), minval=5.0, maxval=50.0)
#        #
#        ## W.l.o.g., normalize the volatilities for a given volatility of the basket.
#        ## It makes plotting the data more convenient.
#        #normalized_vols = (self.weights * vols).reshape((-1, 1))
#        #v = jnp.sqrt(jnp.linalg.multi_dot([normalized_vols.T, correlated_samples, normalized_vols]).reshape(1))
#        #vols = vols * self.vol_basket / v
#        #
#        #diag_v = jnp.diag(vols)
#        #cov = jnp.linalg.multi_dot([diag_v, correlated_samples, diag_v])
#        #key, subkey = jrandom.split(key)
#        
#
#
#        # simulations using fixed cov and seed for paths
#        chol = jnp.linalg.cholesky(cov) * jnp.sqrt(t_delta)
#        # increase vols for simulation of xs so we have more samples in the wings
#        chol_0 = chol * self.vol_mult * jnp.sqrt(self.t_exposure / t_delta)
#
#        #normal_samples = jrandom.normal(subkey, shape=(2, n_samples, self.n_dims))
#        #paths_0 = normal_samples[0] @ chol_0.T
#        #paths_1 = normal_samples[1] @ chol.T
#
#
#        # fresh batch key for S0, fixed key for paths
#        normals_x = jrandom.normal(key, shape=(n_samples, self.n_dims))
#        paths_0 = normals_x @ chol_0.T
#        spots_1 = spots_0 + paths_0
#        
#        normal_samples_paths_1 = jrandom.normal(self.key_train, shape=(n_paths, self.n_dims))
#        paths_1 = normal_samples_paths_1 @ chol.T
#
#        if self.use_antithetic:
#            payoff_fn = Bachelier.antithetic_payoff
#        else:
#            payoff_fn = Bachelier.payoff
#
#        payoff_fn = partial(payoff_fn, weights=self.weights, strike_price=self.strike_price)
#
#
#        # TODO vectorize over spots_1 and average paths?
#
#
#        payoffs_vjp, vjp_fn = jax.vjp(payoff_fn, spots_1, paths_1)
#        differentials_vjp = vjp_fn(jnp.ones(payoffs_vjp.shape))[0]
#
#        return DifferentialData(
#            order = 1,
#            x = spots_1,
#            y = payoffs_vjp,
#            dy = differentials_vjp
#        )
#




    def get_test_set(self, n_samples:int, order:int) -> DifferentialData:
        return self.analytic(n_samples, order=order)



    def analytic(self, n_samples, minval=0.5, maxval=1.5, order=2) -> DifferentialData:
        """TODO: ."""

        # adjust lower and upper for dimension
        adj = 1 + 0.5 * jnp.sqrt((self.n_dims - 1) * (maxval - minval) / 12)
        adj_lower = 1.0 - (1.0 - minval) * adj
        adj_upper = 1.0 + (maxval - 1.0) * adj

        # draw random spots within range
        self.key_test, subkey = jrandom.split(self.key_test)
        spots = jrandom.uniform(subkey, shape=(n_samples, self.n_dims), minval=adj_lower, maxval=adj_upper)
        
        baskets = jnp.dot(spots, self.weights).reshape((-1, 1))
        time_to_maturity = self.t_maturity - self.t_exposure
        
        prices = Bachelier.Call.price(baskets, self.strike_price, self.vol_basket, time_to_maturity)
        prices = prices.reshape((-1,))
        
        # in analytical solution we directly compute greeks w.r.t. the basket price
        deltas = Bachelier.Call.delta(baskets, self.strike_price, self.vol_basket, time_to_maturity)
        deltas = deltas @ self.weights.reshape((1, -1)) # (batch, d) 
        
        gammas = None
        speeds = None
        if order >= 2:
            gammas = Bachelier.Call.gamma(baskets, self.strike_price, self.vol_basket, time_to_maturity)
            w2 = jnp.outer(self.weights, self.weights)
            gammas = gammas.reshape(-1, 1, 1) * w2 # (batch, d, d)
        if order >= 3:
            speeds = Bachelier.Call.speed(baskets, self.strike_price, self.vol_basket, time_to_maturity)
            w3 = jnp.einsum('i,j,k->ijk', self.weights, self.weights, self.weights)  
            speeds = speeds.reshape(-1, 1, 1, 1) * w3      # (batch, d, d, d)
        if order >= 4:
            raise ValueError("Differential Data for order >= 4 not supported")

        return DifferentialData(
            order = order,
            x = spots,
            y = prices,
            dy = deltas,
            ddy = gammas,
            dddy = speeds
        ) 





    # Credit Neil Kichler
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
            #print("got spot shape: ", spot.shape)
            
            sqrt_t = jnp.sqrt(t)
            d = (spot - strike) / (vol * sqrt_t)
            normal_cdf_d = jstats.norm.cdf(d)
            normal_pdf_d = jstats.norm.pdf(d)
            price = vol * sqrt_t * (d * normal_cdf_d + normal_pdf_d)
            #print("shape call.price: ", price.shape)
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
        def speed(spot, strike, vol, t):
            # TODO double check that this is correct
            """Third derivative wrt spot (a.k.a. speed)."""
            d = (spot - strike) / (vol * jnp.sqrt(t))
            return - d * jstats.norm.pdf(d) / (vol**2 * t)


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
            speed = call.speed(spot, strike, vol, t)
            return deltas, gammas, speed
        

    






    def visualize_data(self, dataset: DifferentialData, name: str):

        x = dataset.x
        y = dataset.y
        dy = dataset.dy
        ddy = dataset.ddy
        if dataset.order >= 2:
            # project back onto basket weights
            w = self.weights                      
            ddy = jnp.einsum('bij,i,j->b', ddy, w, w) / ((w @ w) ** 2)  # (b,)
        baskets = jnp.dot(x, self.weights).reshape((-1, 1))

        
        # Create a single figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the first subplot
        axes[0].plot(baskets, y, '.', markersize=1)
        axes[0].set_title(f"Values {name}")

        # Plot the second subplot
        dydx_idx = 0
        axes[1].plot(baskets, dy[:, dydx_idx], '.', markersize=1)
        axes[1].set_title(f"Differentials {name}")

        if dataset.order >= 2 and ddy is not None:
            # Calculate and plot gammas in the third subplot
            #pred_gammas = jnp.sum(pred_ddyddx, axis=(1, 2))
            axes[2].plot(baskets, ddy, '.', markersize=1)
            axes[2].set_title(f"Gammas {name}")

        # Adjust the layout and save the figure to a PDF file
        plt.tight_layout()
        plt.show()
        


    # visualize model predictions
    def plot_eval(self, pred_y, pred_dydx, pred_ddyddx, test_ds: DifferentialData):


        baskets = jnp.dot(test_ds.x, self.weights).reshape((-1, 1))
        y_test = test_ds.y
        dydx_test = test_ds.dy
        gammas = test_ds.ddy

        pred_y = pred_y[:, jnp.newaxis]

        # Create a single figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the first subplot
        axes[0].plot(baskets, pred_y, '.', markersize=1)
        axes[0].plot(baskets, y_test, '.', markersize=1)
        axes[0].legend(['Pred Price', 'True Price'], loc='upper left')
        axes[0].set_title(f"Values \n rmse: {rmse(pred_y, y_test)}")

        # Plot the second subplot
        dydx_idx = 0
        axes[1].plot(baskets, pred_dydx[:, dydx_idx], '.', markersize=1)
        axes[1].plot(baskets, dydx_test[:, dydx_idx], '.', markersize=1)
        axes[1].legend(['Pred Delta', 'True Delta'], loc='upper left')
        axes[1].set_title(f"Differentials\nrmse: {rmse(pred_dydx, dydx_test)}")

        # Calculate and plot gammas in the third subplot
        pred_gammas = jnp.sum(pred_ddyddx, axis=(1, 2))
        axes[2].plot(baskets, pred_gammas, '.', markersize=1, label='Pred')
        axes[2].plot(baskets, gammas, '.', markersize=1, label='True')
        axes[2].legend()
        axes[2].set_title(f"Gammas\nrmse: {rmse(pred_gammas, gammas)}")

        # Adjust the layout and save the figure to a PDF file
        plt.tight_layout()
        plt.show()
        #now = datetime.datetime.now()
        #fig.savefig(f'result/eval_ml_{now}.pdf', bbox_inches='tight')

