"""
TODO
"""
from jaxtyping import Array, Float, PRNGKeyArray

import jax
import jax.numpy as jnp
import jax.random as jrandom

import jax.numpy as jnp

from diff_ml.utils import Range, normalize
from diff_ml.typing import DifferentialData, Scalar
from diff_ml.reference_models.reference_model_class import ReferenceModel    
from diff_ml.plotting import plot_3d_differential_data


class Heston(ReferenceModel):

    key_test: PRNGKeyArray
    key_train: PRNGKeyArray
    n_dims: int

    # for basket
    basket_dim: int
    basket_weights: Float[Array, "basket dim"]

    spot_range:Range = Range(50.0, 150.0)
    vol_range: Range = Range(0.01, 0.1)
  
    K: float = 100.0        # Strike price
    r: float = 0.00  #0.05  # Risk-free rate
    T: float = 1.0   #5.0   # n years until expiry, i.e. T2 - T1
    rho: float = -0.3       # Correlation of asset and volatility
    kappa: float = 1.00 #2.0# Mean-reversion rate
    theta: float = 0.09     # Long run average volatility
    xi: float = 1.00        # Volatility of volatility

    # for semi-closed-form solution
    num_w: int = 512
    w_max: float = 100.0

    x_mean: Float[Array, "basket_dim 2"]
    x_std:  Float[Array, "basket_dim 2"]
    y_mean: float
    y_std: float


    def __init__(
            self,
            key: PRNGKeyArray,
            basket_dim: int,
            basket_weights: Float[Array, "basket dim"],
            ):
        self.key_test, self.key_train, y_sample_key = jax.random.split(key, 3)
        self.basket_dim = basket_dim
        self.basket_weights = basket_weights
        self.n_dims = 2*self.basket_dim
        
        # x mean std, y mean std for normalization 
        # Uniform[a,b] has mean (a+b)/2, var (b−a)**2/12
        mean_spot = 0.5 * (self.spot_range.minval + self.spot_range.maxval)
        std_spot = jnp.sqrt((self.spot_range.maxval - self.spot_range.minval) ** 2 / 12)
        mean_vol  = 0.5 * (self.vol_range.minval  + self.vol_range.maxval)
        std_vol  = jnp.sqrt((self.vol_range.maxval  - self.vol_range.minval ) ** 2 / 12)

        x_means = jnp.stack([
            jnp.full(self.basket_dim, mean_spot),
            jnp.full(self.basket_dim, mean_vol),
        ], axis=1)  # shape = (basket_dim, 2)
        x_stds = jnp.stack([
            jnp.full(self.basket_dim, std_spot),
            jnp.full(self.basket_dim, std_vol),
        ], axis=1)

        self.x_mean = x_means
        self.x_std = x_stds    

        sample_ys = self.sample_raw_ys(key=y_sample_key, n_samples=1024)
        self.y_mean = jnp.mean(sample_ys).item()
        self.y_std = jnp.std(sample_ys).item()      
        
        
        
    


    def closed_form_price(self, S0, v0) -> Scalar:
        """
        Compute the Heston model European call price using the semi-closed-form solution using numerical integration in JAX.

        Math from:  https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf
        and form:   https://xilinx.github.io/Vitis_Libraries/quantitative_finance/2019.2/methods/cf-ht.html#chrso2014
        
        Parameters:
        - S0: initial spot price
        - v0: initial variance
       
        Returns:
        - call option price under Heston model
        """

        sigma = self.xi # volatility of volatility

        # integration grid
        w = jnp.linspace(0.0, self.w_max, self.num_w)[1:]  # avoid w=0

        # Heston characteristic function components
        a = self.kappa * self.theta
        def C_D(wi):
            alpha = -0.5 * wi**2 - 0.5j * wi
            beta  = alpha - 1j * self.rho * sigma * wi
            gamma = 0.5 * sigma**2

            h = jnp.sqrt(beta**2 - 4 * alpha * gamma)
            rp = (beta + h) / (sigma**2)
            rm = (beta - h) / (sigma**2)
            g = rm / rp

            exp_neg_ht = jnp.exp(-h * self.T)
            C = a * (rm * self.T - (2.0 / sigma**2) * jnp.log((1 - g * exp_neg_ht) / (1 - g)))
            D = rm * (1 - exp_neg_ht) / (1 - g * exp_neg_ht)
            return C, D

        # vectorize C and D over w
        C_vec, D_vec = jax.vmap(C_D)(w)

        # characteristic function Psi
        log_term = jnp.log(S0) + self.r * self.T
        def Psi(wi, Ci, Di):
            return jnp.exp(Ci * self.theta + Di * v0 + 1j * wi * log_term)

        Psi_w      = Psi(w,      C_vec, D_vec)
        Psi_w_minus_i = Psi(w - 1j, *jax.vmap(C_D)(w - 1j))
        Psi_minus_i = Psi(-1j, *C_D(-1j))

        # integrands for Pi1 and Pi2
        integrand1 = jnp.real(jnp.exp(-1j * w * jnp.log(self.K)) * Psi_w_minus_i / (1j * w * Psi_minus_i))
        integrand2 = jnp.real(jnp.exp(-1j * w * jnp.log(self.K)) * Psi_w      / (1j * w))

        # numerical integration via the trapezoidal rule
        Pi1 = 0.5 + (1.0 / jnp.pi) * jnp.trapezoid(integrand1, w)
        Pi2 = 0.5 + (1.0 / jnp.pi) * jnp.trapezoid(integrand2, w)

        # final price
        price = S0 * Pi1 - jnp.exp(-self.r * self.T) * self.K * Pi2
        return price
    



    def closed_form_basket_price(self, basket_S0s, basket_v0s) -> Scalar:
        # vectorize single‐asset pricer over the basket axis
        prices = jax.vmap(self.closed_form_price)(basket_S0s, basket_v0s)
        # Introduce interactions
        interaction_term = jnp.sum((basket_S0s - jnp.mean(basket_S0s)) * (basket_v0s - jnp.mean(basket_v0s)))
        return jnp.dot(self.basket_weights, prices) #+ 0.2 * interaction_term
    

    def basket_price_x_flat(self, x_flat):
        z = x_flat.reshape(self.basket_dim, 2)
        S0s = z[:, 0]
        v0s = z[:, 1]
        return self.closed_form_basket_price(S0s, v0s)
       
    
    def normalized_wrapper(self, x_flat_normalized) -> Scalar:
        # un-normalize inputs x
        x_normalized_unflat = x_flat_normalized.reshape(self.basket_dim, 2)
        x_raw_unflat = x_normalized_unflat * self.x_std + self.x_mean
        x_raw_flat = x_raw_unflat.reshape(self.n_dims)
        # call in raw space
        y = self.basket_price_x_flat(x_raw_flat)
        # re-normalize y
        y_normalized = (y - self.y_mean) / self.y_std
        return y_normalized    

    def reference_fn(self):
        #return self.basket_price_x_flat
        return self.normalized_wrapper





    def get_test_set(self, n_samples: int, order:int) -> DifferentialData:
        return self.sample(self.key_test, n_samples, order)


    def sample_initial_states(self, key: PRNGKeyArray, n_samples: int):
        minvals=jnp.array([self.spot_range.minval, self.vol_range.minval])
        maxvals=jnp.array([self.spot_range.maxval, self.vol_range.maxval])
        
        initial_states = jrandom.uniform(
            key, 
            shape=(n_samples, self.basket_dim, 2), 
            minval=minvals, 
            maxval=maxvals
        ) # (batch, n_dims, 2)
        S0s = initial_states[..., 0]    # (batch, n_dims)
        v0s = initial_states[..., 1]     # (batch, n_dims)
        return S0s, v0s
    

    def sample_raw_ys(self, key: PRNGKeyArray, n_samples: int):
        S0s, v0s = self.sample_initial_states(key, n_samples)
        ys = jax.vmap(self.closed_form_basket_price)(S0s, v0s)
        return ys



    def sample(self, key: PRNGKeyArray, n_samples: int, order: int=1) -> DifferentialData:    
        
        S0s, v0s = self.sample_initial_states(key, n_samples)
        
        value_and_grad_fn = jax.value_and_grad(self.closed_form_basket_price, argnums=(0,1))
        y, (dS0s, dV0s) = jax.vmap(value_and_grad_fn)(S0s, v0s)
        dydx = jnp.stack([dS0s, dV0s], axis=-1)
        
        initial_states = jnp.stack([S0s, v0s], axis=-1)  

        x_normalized = normalize(initial_states, self.x_mean, self.x_std)
        y_normalized = normalize(y, self.y_mean, self.y_std)
        dydx_normalized = dydx * (self.x_std / self.y_std)[None, ...]

    
        # flatten before returning    
        x_flat = initial_states.reshape(n_samples, -1)
        x_normalized_flat = x_normalized.reshape(n_samples, -1)
        dydx_normalized_flat = dydx_normalized.reshape(n_samples, -1)


        ddy = None
        dddy = None

        if order >= 2:
            
            H_full = jax.vmap(jax.hessian(self.basket_price_x_flat))(x_flat) # (batch, basket*2, basket*2)
            
            x_std_flat = self.x_std.reshape(-1)               # (d,)
            # build the (d x d) scaling matrix
            scale = jnp.outer(x_std_flat, x_std_flat) / self.y_std
            # broadcast over the batch dimension:
            ddyddx_normalized = H_full * scale[None, :, :]
            ddy = ddyddx_normalized
            
            ## TODO make this also in flat space
            ## TODO then x_mean and std also just flat
            ## working with un-flat x_std (basket, 2)
            #H_blocks = H_full.reshape(
            #    n_samples,
            #    self.basket_dim, 2,
            #    self.basket_dim, 2
            #) # (batch, basket, 2 basket, 2)
            #
            #scale_full = (self.x_std[:, :, None, None] * self.x_std[None, None, :, :]) / self.y_std
            #
            ## 4) broadcast and apply
            #H_blocks_norm = H_blocks * scale_full[None, ...]
            ## H_blocks_norm: (batch, n, 2, n, 2)
            ##norm_values["H_scale_full"] = scale_full
            #ddyddx_normalized = H_blocks_norm
            #ddy = ddyddx_normalized


        if order >= 3:
            # 3rd order
            dddydddx = jax.vmap(jax.jacfwd(jax.hessian(self.basket_price_x_flat)))(x_flat) # (b, basket*2, basket*2, basket*2)
            #jax.debug.print("dddydddx shape: {dddydddx_shape}", dddydddx_shape=dddydddx.shape)
            
            # working with flattened x_std
            x_std_flat = self.x_std.reshape(-1)                 # (d,)
            scale3 = (x_std_flat[:, None, None] *
                      x_std_flat[None, :, None] *
                      x_std_flat[None, None, :]) / self.y_std   # (d, d, d)
            
            # apply normalization broadcasted over batch dim
            dddydddx_normalized = dddydddx * scale3[None, ...]  # (batch, d, d, d)
            dddy = dddydddx_normalized


        return DifferentialData(
            order = order,
            x = x_normalized_flat,
            y = y_normalized,
            dy = dydx_normalized_flat,
            ddy = ddy,
            dddy = dddy
        )




    def visualize_data(self, dataset: DifferentialData, name: str):
        
        # TODO multiply with basket weights before as in Bachelier?

        plot_3d_differential_data(
            dataset=dataset,
            name=name,
            x1_index=0,
            x2_index=1,
            x1_name="asset0_S0",
            x2_name="asset0_v0"
        )
