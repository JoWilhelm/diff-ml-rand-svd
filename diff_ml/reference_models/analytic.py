"""
TODO
"""
import jax
import jax.numpy as jnp

from diff_ml.utils import normalize
from diff_ml.plotting import plot_3d_differential_data

from typing import Callable
from jaxtyping import Array, Float, PRNGKeyArray

from diff_ml.reference_models.reference_model_class import ReferenceModel

from diff_ml.typing import DifferentialData, Scalar






class Analytic(ReferenceModel):

    key_test: PRNGKeyArray
    key_train: PRNGKeyArray
    n_dims: int

    type: str
    
    min_x: float
    max_x: float
    
    x_mean: Float[Array, "d"]
    x_std: Float[Array, "d"]
    y_mean: float
    y_std: float
    


    def __init__(self, key: PRNGKeyArray, type: str, d: int, min_x: float, max_x: float):

        self.key_test, self.key_train, y_sample_key = jax.random.split(key, 3)

        self.type = type
        self.n_dims = d
        self.min_x = min_x
        self.max_x = max_x

        # set x mean std, y mean std for normalization
        m = 0.5 * (self.min_x + self.max_x)
        s = jnp.sqrt((self.max_x - self.min_x) ** 2 / 12)
        self.x_mean = jnp.full((self.n_dims,), m)
        self.x_std  = jnp.full((self.n_dims,), s)
        # sample some ys to get mean and std
        sample_ys = self.sample_raw_ys(key=y_sample_key, n_samples=1024)
        self.y_mean = jnp.mean(sample_ys).item()
        self.y_std = jnp.std(sample_ys).item()
        


        
    # rotated hyper ellipsoid
    # d constant, no dependence on x
    # https://www.sfu.ca/~ssurjano/rothyp.html
    def rotated_hyper_ellipsoid(self, x) -> Scalar:
        d = x.shape[-1]
        weights = jnp.arange(d, 0, -1)  # (d,)
        return jnp.sum(weights * x**2, axis=-1)
    

    # Rastrigin function
    # d varying, only maginutude depends on x
    # https://en.wikipedia.org/wiki/
    def rastrigin(self, x, A: float = 10.0) -> Scalar:
        two_pi_x = 2.0 * jnp.pi * x
        return A * x.shape[-1] + jnp.sum(x**2 - A * jnp.cos(two_pi_x), axis=-1)

    
    # Rosenbrock function
    # strong decay, dependece on x
    #https://en.wikipedia.org/wiki/Rosenbrock_function
    def rosenbrock(self, x) -> Scalar:
        x_i   = x[..., :-1]
        x_ip1 = x[..., 1:]
        return jnp.sum(100.0 * (x_ip1 - x_i**2)**2 + (1.0 - x_i)**2, axis=-1)


    # Ackley function
    # moderate decay, dependence on x
    # https://www.sfu.ca/~ssurjano/ackley.html
    def ackley(self, x, a: float = 20.0, b: float = 0.2, c: float = 2.0 * jnp.pi) -> Scalar:
        # mean of squared components
        msq = jnp.mean(x**2, axis=-1)
        term1 = -a * jnp.exp(-b * jnp.sqrt(msq + 1e-12)) 
        # mean of cos(c * x_i)
        mcos = jnp.mean(jnp.cos(c * x), axis=-1)
        term2 = -jnp.exp(mcos)
        return term1 + term2 + a + jnp.e





    def type_fn(self) -> Callable[[Float[Array, "d"]], Scalar]:
        if self.type == "RHE":
            return self.rotated_hyper_ellipsoid
        elif self.type == "Rastrigin":
            return self.rastrigin
        elif self.type == "Rosenbrock":
            return self.rosenbrock
        elif self.type == "Ackley":
            return self.ackley
        else:
            raise ValueError(f"Unknown function type: {self.type}")



    def normalized_wrapper(self, x_normalized) -> Scalar:
        # un-normalize inputs x
        x_raw = x_normalized * self.x_std + self.x_mean
        # call in raw space
        y = self.type_fn()(x_raw)
        # re-normalize y
        y_normalized = (y - self.y_mean) / self.y_std
        return y_normalized    



    def reference_fn(self) -> Callable[[Float[Array, "d"]], Scalar]:
        return self.normalized_wrapper



    def get_test_set(self, n_samples: int, order: int) -> DifferentialData:
        return self.sample(self.key_test, n_samples, order=order)



    def sample_raw_ys(self, key, n_samples):
        x = jax.random.uniform(key, (n_samples, self.n_dims),minval=self.min_x, maxval=self.max_x)
        ys = jax.vmap(self.type_fn())(x)
        return ys



    def sample(self, key: PRNGKeyArray, n_samples=256, order=1) -> DifferentialData:    
        
        # uniformly sample inputs over [minx, maxx] range
        x = jax.random.uniform(key, (n_samples, self.n_dims),minval=self.min_x, maxval=self.max_x)

        # y aand dy
        value_and_grad_fn = jax.value_and_grad(self.type_fn())
        y, dydx = jax.vmap(value_and_grad_fn)(x)
                
        x_normalized = normalize(x, self.x_mean, self.x_std)
        y_normalized = normalize(y, self.y_mean, self.y_std)
        dydx_normalized = dydx * (self.x_std / self.y_std)#[None, ...]
        
        ddy = None
        dddy = None

        if order >= 2:
            # 2nd order
            ddyddx = jax.vmap(jax.hessian(self.type_fn()))(x)
            # build the (d x d) scaling matrix
            scale = jnp.outer(self.x_std, self.x_std) / self.y_std
            # broadcast over the batch dimension:
            ddyddx_normalized = ddyddx * scale[None, :, :]
            ddy = ddyddx_normalized
        if order >= 3:
            # 3rd order
            dddydddx = jax.vmap(jax.jacfwd(jax.hessian(self.type_fn())))(x)
            # build (dxdxd) scale tensor
            scale3 = (self.x_std[:, None, None] *
                      self.x_std[None, :, None] *
                      self.x_std[None, None, :]) / self.y_std
            # broadcast over batch
            dddydddx_normalized = dddydddx * scale3[None, :, :, :]
            dddy = dddydddx_normalized
        if order >= 4:
            raise ValueError("Differential Data for order >= 4 not supported")


        return DifferentialData(
            order=order,
            x=x_normalized,
            y=y_normalized,
            dy=dydx_normalized,
            ddy=ddy,
            dddy=dddy,
        )





    def visualize_data(self, dataset: DifferentialData, name: str):
        plot_3d_differential_data(
            dataset=dataset,
            name=name,
            x1_index=0,
            x2_index=1,
            x1_name="x1",
            x2_name="x2"
        )
