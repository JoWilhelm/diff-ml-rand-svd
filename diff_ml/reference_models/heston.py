from typing import Final
from dataclasses import field
from typing import Tuple
from jaxtyping import Array, Float, PRNGKeyArray


import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jrandom

from typing_extensions import TypeAlias



import jax.numpy as jnp

Data: TypeAlias = dict[str, Float[Array, "n_samples ..."]]
from diff_ml.utils import Range, normalize

    
class Heston(eqx.Module):

    key: PRNGKeyArray
    
    # for basket
    basket_dim: Final[int]
    basket_weights: Float[Array, "basket dim"]


    spot_range:Range = Range(50.0, 150.0)
    vol_range: Range = Range(0.01, 0.1)

    mc_time_steps: int = 1024 # number of time steps for the Monte Carlo simulation
     
    K: float = 100.0        # Strike price
    r: float = 0.05         # Risk-free rate
    T: float = 5.0          # n years until expiry, i.e. T2 - T1
    rho: float = -0.3       # Correlation of asset and volatility
    kappa: float = 2.00     # Mean-reversion rate
    theta: float = 0.09     # Long run average volatility
    xi: float = 1.00        # Volatility of volatility


    # for semi-closed-form solution
    num_w: int = 512
    w_max: float = 100.0

    was_normalized = False
    y_mean: float = None
    y_std: float = None


    # derived, init‑only fields
    x_mean: Float[Array, "basket_dim 2"] = field(init=False)
    x_std:  Float[Array, "basket_dim 2"] = field(init=False)
    n_dims: int = field(init=False)
    un_flattened_shape: Tuple = field(init=False)

    def set_y_std(self, y_std):
        object.__setattr__(self, "y_std", y_std)
        
    def set_y_mean(self, y_mean):
        object.__setattr__(self, "y_mean", y_mean)

    # this is ugly. Build a propoer init function?
    def __post_init__(self):

        # Uniform[a,b] has mean (a+b)/2, var (b−a)**2/12
        mean_spot = 0.5 * (self.spot_range.minval + self.spot_range.maxval)
        std_spot = jnp.sqrt((self.spot_range.maxval - self.spot_range.minval) ** 2 / 12)
        mean_vol  = 0.5 * (self.vol_range.minval  + self.vol_range.maxval)
        std_vol  = jnp.sqrt((self.vol_range.maxval  - self.vol_range.minval ) ** 2 / 12)

        means = jnp.stack([
            jnp.full(self.basket_dim, mean_spot),
            jnp.full(self.basket_dim, mean_vol),
        ], axis=1)  # shape = (basket_dim, 2)

        stds = jnp.stack([
            jnp.full(self.basket_dim, std_spot),
            jnp.full(self.basket_dim, std_vol),
        ], axis=1)

        # Bypass frozen guard
        object.__setattr__(self, "x_mean", means)
        object.__setattr__(self, "x_std",  stds)
        
        
        n_dims = 2*self.basket_dim
        object.__setattr__(self, "n_dims", n_dims)

        
        un_flattened_shape = (self.basket_dim, 2)
        object.__setattr__(self, "un_flattened_shape", un_flattened_shape)
 
    
    


    def closed_form_price(self, S0, v0):
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
    

    def closed_form_basket_price(self, basket_S0s, basket_v0s):
        # vectorize single‐asset pricer over the basket axis
        prices = jax.vmap(self.closed_form_price)(basket_S0s, basket_v0s)
        
        # Introduce interactions
        interaction_term = jnp.sum((basket_S0s - jnp.mean(basket_S0s)) * (basket_v0s - jnp.mean(basket_v0s)))
    
        return jnp.dot(self.basket_weights, prices) + 0.2 * interaction_term
    

    def basket_price_x_flat(self, x_flat):
        d = self.basket_dim
        
        #S0s = x_flat[:d]        # first n_dims entries
        #v0s = x_flat[d:]        # next n_dims entries
        z = x_flat.reshape(self.basket_dim, 2)   # interleaved layout
        S0s = z[:, 0]
        v0s = z[:, 1]
        return self.closed_form_basket_price(S0s, v0s)
       
    
    def normalized_reference_fn(self, x_flat_normalized):
        # un-normalize x
        x_normalized_unflat = x_flat_normalized.reshape(self.un_flattened_shape)
        x_raw_unflat = x_normalized_unflat * self.x_std + self.x_mean
        x_flat_unnormalized = x_raw_unflat.reshape(self.n_dims)
        
        y = self.basket_price_x_flat(x_flat_unnormalized)

        # re-normalize y
        y_normalized = (y - self.y_mean) / self.y_std

        return y_normalized    

    def reference_fn(self, *args):
        #return self.basket_price_x_flat
        return self.normalized_reference_fn


    def get_test_set(self, n_samples):
        return self.sample(self.key, n_samples, is_test=True)


    def sample(self, key, n_samples=256, is_test=False):    
        
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
    



        value_and_grad_fn = jax.value_and_grad(self.closed_form_basket_price, argnums=(0,1))
        y, (dS0s, dV0s) = jax.vmap(value_and_grad_fn)(S0s, v0s)
        dydx = jnp.stack([dS0s, dV0s], axis=-1)
        # x:     (batch, n_dims, 2)
        # y:     (batch,)
        # dydx:  (batch, n_dims, 2)
        #print("x shape: ", initial_states.shape)
        #print("y shape: ", y.shape)
        #print("dydx shape: ", dydx.shape)

        if is_test: 
            # use test set y as uniform reference for future batches   
            self.set_y_mean(jnp.mean(y))
            self.set_y_std(jnp.std(y))      
        
        
        x_normalized = normalize(initial_states, self.x_mean, self.x_std)
        y_normalized = normalize(y, self.y_mean, self.y_std)
        dydx_normalized = dydx * (self.x_std / self.y_std)[None, ...]

        #norm_values = {
        #    "x_mean": x_mean,
        #    "x_std": x_std,
        #    "y_mean": y_mean,
        #    "y_std": y_std,
        #}

    
        ## flatten before returning    
        x_flat = initial_states.reshape(initial_states.shape[0], -1)
        dydx_flat = dydx.reshape(dydx.shape[0], -1)
        x_normalized_flat = x_normalized.reshape(x_normalized.shape[0], -1)
        x_raw_flat = initial_states.reshape(initial_states.shape[0], -1)
        dydx_normalized_flat = dydx_normalized.reshape(dydx_normalized.shape[0], -1)

        if not is_test:
            # TODO move normalization out of sample function and call in loss function
            
            #return x_flat, y, dydx_flat, None
            return x_normalized_flat, y_normalized, dydx_normalized_flat, None



        if is_test:
            # use jax.hessian to compute the second derivatives
            
            x_flat = initial_states.reshape(n_samples, 2*self.basket_dim)

            H_full = jax.vmap(jax.hessian(self.basket_price_x_flat))(x_flat)

            H_blocks = H_full.reshape(
                n_samples,
                self.basket_dim, 2,
                self.basket_dim, 2
            ) # (batch, basket, 2 basket, 2)

            scale_full = (self.x_std[:, :, None, None] * self.x_std[None, None, :, :]) / self.y_std

            # 4) broadcast and apply
            H_blocks_norm = H_blocks * scale_full[None, ...]
            # H_blocks_norm: (batch, n, 2, n, 2)
            #norm_values["H_scale_full"] = scale_full
            ddyddx_normalized = H_blocks_norm




            ## 3rd order
            #dddydddx = jax.vmap(jax.jacfwd(jax.hessian(self.basket_price_x_flat)))(x_flat) # (b, 2, 2, 2)
            ##jax.debug.print("dddydddx shape: {dddydddx_shape}", dddydddx_shape=dddydddx.shape)
            #
            #
            #x_std_flat = self.x_std.reshape(-1)               # (d,)
            ## scale3[i,j,k] = x_std[i] * x_std[j] * x_std[k] / y_std
            #scale3 = (x_std_flat[:, None, None] *
            #          x_std_flat[None, :, None] *
            #          x_std_flat[None, None, :]) / self.y_std   # (d, d, d)
            #
            ## broadcast over batch and apply
            #dddydddx_normalized = dddydddx * scale3[None, ...]  # (batch, d, d, d)
            #
            ### build (d×d×d) scale tensor
            ##scale3 = 
            ### broadcast over batch
            ##dddydddx_normalized = dddydddx * scale3[None, ...]




            #return x_flat, y, dydx_flat, H_blocks, None
            return x_normalized_flat, y_normalized, dydx_normalized_flat, ddyddx_normalized, None #dddydddx_normalized






    def visualize_dataset(self, dataset, name, is_second_order):
        # visulaize the test set

        if is_second_order:
            x, y, dydx, ddyddx, _ = dataset
        else:
            x, y, dydx, _ = dataset
        
        x = x.reshape(x.shape[0], *self.un_flattened_shape)
        dydx = dydx.reshape(dydx.shape[0], *self.un_flattened_shape)


        
        # selecting first basket dimension
        basket_i = 0

        x = x[:, basket_i, :]  # (batch_size, 2)
        y = y  # (batch_size, )
        dydx = dydx[:, basket_i, :]  # (batch_size, 2)
        
        if is_second_order:
            ddyddx = jnp.stack([ddyddx[:,i,: ,i,:] for i in range(self.basket_dim)], axis=1)
            ddyddx = ddyddx[:, basket_i, :, :]  # (batch_size, k_probe_directions, 2, 2)

        ## average over the basket dimension
        #x = x.mean(axis=1)
        #y = y
        #dydx = dydx.mean(axis=1) 
        #ddyddx = ddyddx.mean(axis=2)

        print("shapes without basket dimension:")
        print("x shape: ", x.shape)
        print("y shape: ", y.shape)
        print("dydx shape: ", dydx.shape)
        if is_second_order:
            print("ddyddx shape: ", ddyddx.shape)




        xs = x[..., 0]
        ys = x[..., 1]


        fig_payoff = plot_3d_data(xs, ys, y, x1_label="$S0$", x2_label="$v0$", y_label="y" , title=f"payoff\n{name} target")

        fig_dS = plot_3d_data(xs, ys, dydx[:, 0], x1_label="$S0$", x2_label="$v0$", y_label="dydS0" , title=f"1st-order diff payoff - dS\n{name} target")
        fig_dv = plot_3d_data(xs, ys, dydx[:, 1], x1_label="$S0$", x2_label="$v0$", y_label="dydv0" , title=f"1st-order diff payoff - dv\n{name} target")
        


        if is_second_order:
            fig_dS = plot_3d_data(xs, ys, ddyddx[:, 0, 0], x1_label="$S0$", x2_label="$v0$", y_label="ddyddS0S0" , title=f"2nd-order diff payoff - dSdS\n{name} target")
            fig_dv = plot_3d_data(xs, ys, ddyddx[:, 0, 1], x1_label="$S0$", x2_label="$v0$", y_label="ddyddS0v0" , title=f"1st-order diff payoff - dSdv\n{name} target")
            fig_dS = plot_3d_data(xs, ys, ddyddx[:, 1, 0], x1_label="$S0$", x2_label="$v0$", y_label="ddyddv0S0" , title=f"2nd-order diff payoff - dvdS\n{name} target")
            fig_dv = plot_3d_data(xs, ys, ddyddx[:, 1, 1], x1_label="$S0$", x2_label="$v0$", y_label="ddyddv0v0" , title=f"1st-order diff payoff - dvdv\n{name} target")



    def visualize_third(self, x, dddydddx, name):
        # visulaize the test set

        print("ddyddx shape: ", dddydddx.shape)


        xs = x[..., 0]
        ys = x[..., 1]

        fig_dx1dx1dx1 = plot_3d_data(xs, ys, dddydddx[:, 0, 0, 0], x1_label="x1", x2_label="x2", y_label="dddydddx111", title=f"{name} target\ndddydddx111")
        fig_dx1dx1dx2 = plot_3d_data(xs, ys, dddydddx[:, 0, 0, 1], x1_label="x1", x2_label="x2", y_label="dddydddx112", title=f"{name} target\ndddydddx112")
        
        fig_dx1dx2dx1 = plot_3d_data(xs, ys, dddydddx[:, 0, 1, 0], x1_label="x1", x2_label="x2", y_label="dddydddx121", title=f"{name} target\ndddydddx121")
        fig_dx1dx2dx2 = plot_3d_data(xs, ys, dddydddx[:, 0, 1, 1], x1_label="x1", x2_label="x2", y_label="dddydddx122", title=f"{name} target\ndddydddx122")
        

        fig_dx2dx1dx1 = plot_3d_data(xs, ys, dddydddx[:, 1, 0, 0], x1_label="x1", x2_label="x2", y_label="dddydddx211", title=f"{name} target\ndddydddx211")
        fig_dx2dx1dx2 = plot_3d_data(xs, ys, dddydddx[:, 1, 0, 1], x1_label="x1", x2_label="x2", y_label="dddydddx212", title=f"{name} target\ndddydddx212")
        
        fig_dx2dx2dx1 = plot_3d_data(xs, ys, dddydddx[:, 1, 1, 0], x1_label="x1", x2_label="x2", y_label="dddydddx221", title=f"{name} target\ndddydddx221")
        fig_dx2dx2dx2 = plot_3d_data(xs, ys, dddydddx[:, 1, 1, 1], x1_label="x1", x2_label="x2", y_label="dddydddx222", title=f"{name} target\ndddydddx222")
        