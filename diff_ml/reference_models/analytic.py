"""
TODO
"""
import jax
import jax.numpy as jnp

from diff_ml.utils import normalize
from diff_ml.plotting import plot_3d_data


# TODO typing stuff clean
from typing import Tuple, Callable
from jaxtyping import Array, Float, PRNGKeyArray

from reference_models.reference_model_class import ReferenceModel

from diff_ml.typing import Data






class Analytic(ReferenceModel):

    key: PRNGKeyArray
    
    type: str

    n_dims: int
    un_flattened_shape: Tuple    

    min_x: float
    max_x: float
    
    x_mean: Float[Array, "d"]
    x_std: Float[Array, "d"]
    y_mean: float = None
    y_std: float = None
    was_normalized = False
    
    def set_y_std(self, y_std):
        object.__setattr__(self, "y_std", y_std)
    def set_y_mean(self, y_mean):
        object.__setattr__(self, "y_mean", y_mean)
    

    def __init__(self, key: PRNGKeyArray, type: str, d: int, min_x: float, max_x: float):

        self.key = key
        self.type = type
        self.n_dims = d
        self.un_flattened_shape = (d,)
        self.min_x = min_x
        self.max_x = max_x


        m = 0.5 * (self.min_x + self.max_x)
        s = jnp.sqrt((self.max_x - self.min_x) ** 2 / 12)
        self.x_mean = jnp.full((self.n_dims,), m)
        self.x_std  = jnp.full((self.n_dims,), s)
        


    
    

    # rotated hyper ellipsoid
    # d constant, no dependence on x
    # https://www.sfu.ca/~ssurjano/rothyp.html
    def rotated_hyper_ellipsoid(self, x):
        d = x.shape[-1]
        weights = jnp.arange(d, 0, -1)  # (d,)
        return jnp.sum(weights * x**2, axis=-1).item()
    

    # Rastrigin function
    # d varying, only maginutude depends on x
    # https://en.wikipedia.org/wiki/
    def rastrigin(self, x, A: float = 10.0):
        two_pi_x = 2.0 * jnp.pi * x
        return A * x.shape[-1] + jnp.sum(x**2 - A * jnp.cos(two_pi_x), axis=-1)

    



    # Rosenbrock function
    # strong decay, dependece on x
    #https://en.wikipedia.org/wiki/Rosenbrock_function
    def rosenbrock(self, x):
        x_i   = x[..., :-1]
        x_ip1 = x[..., 1:]
        return jnp.sum(100.0 * (x_ip1 - x_i**2)**2 + (1.0 - x_i)**2, axis=-1).item()




    # Ackley function
    # moderate decay, dependence on x
    # https://www.sfu.ca/~ssurjano/ackley.html
    def ackley(self, x, a: float = 20.0, b: float = 0.2, c: float = 2.0 * jnp.pi) -> float:
        # mean of squared components
        msq = jnp.mean(x**2, axis=-1)
        term1 = -a * jnp.exp(-b * jnp.sqrt(msq + 1e-12)) 
        # mean of cos(c * x_i)
        mcos = jnp.mean(jnp.cos(c * x), axis=-1)
        term2 = -jnp.exp(mcos)
        return (term1 + term2 + a + jnp.e).item()





    def type_fn(self) -> Callable[[Float[Array, "d"]], float]:
        # Return a bound function (callable) that maps x -> scalar (...):
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




    def normalized_wrapper(self, x_flat_normalized) -> float:

        # un-normalize x
        x_normalized_unflat = x_flat_normalized.reshape(self.un_flattened_shape)
        x_raw_unflat = x_normalized_unflat * self.x_std + self.x_mean
        x_flat_unnormalized = x_raw_unflat.reshape(self.n_dims)
        
        # call in raw space
        y = self.type_fn()(x_flat_unnormalized)
        
        # re-normalize y
        y_normalized = (y - self.y_mean) / self.y_std

        return y_normalized    



    def reference_fn(self) -> Callable[[Float[Array, "d"]], float]:
        return self.normalized_wrapper




    def get_test_set(self, n_samples: int) -> Data:
        return self.sample(self.key, n_samples, higher_order=True)



    def sample(self, key: PRNGKeyArray, n_samples=256, higher_order=False) -> Data:    
        

        x = jax.random.uniform(key, (n_samples, self.n_dims),minval=self.min_x, maxval=self.max_x)


        value_and_grad_fn = jax.value_and_grad(self.type_fn())
        y, dydx = jax.vmap(value_and_grad_fn)(x)
        

        if higher_order: 
            # use test set y as uniform reference for future batches   
            self.set_y_mean(jnp.mean(y))
            self.set_y_std(jnp.std(y))      
        
        
        x_normalized = normalize(x, self.x_mean, self.x_std)
        y_normalized = normalize(y, self.y_mean, self.y_std)
        dydx_normalized = dydx * (self.x_std / self.y_std)#[None, ...]
        

        if not higher_order:
            return {
                "x": x_normalized,
                "y": y_normalized,
                "dy": dydx_normalized,
            }

        if higher_order:
            
            # 2nd order
            ddyddx = jax.vmap(jax.hessian(self.type_fn()))(x)
            # build the (d × d) scaling matrix
            scale = jnp.outer(self.x_std, self.x_std) / self.y_std
            # broadcast over the batch dimension:
            ddyddx_normalized = ddyddx * scale[None, :, :]



            ## 3rd order
            #dddydddx = jax.vmap(jax.jacfwd(jax.hessian(self.type_fn())))(x)
            ## build (d×d×d) scale tensor
            #scale3 = (self.x_std[:, None, None] *
            #          self.x_std[None, :, None] *
            #          self.x_std[None, None, :]) / self.y_std
            ## broadcast over batch
            #dddydddx_normalized = dddydddx * scale3[None, :, :, :]



            #return x_normalized, y_normalized, dydx_normalized, ddyddx_normalized, None#dddydddx_normalized
            return {
                "x": x_normalized,
                "y": y_normalized,
                "dy": dydx_normalized,
                "ddy": ddyddx_normalized,
                #"dddy": dddydddx_normalized,
            }













    def visualize_dataset(self, dataset, name, is_second_order):
            # visulaize the test set

            if is_second_order:
                x, y, dydx, ddyddx, _ = dataset
            else:
                x, y, dydx, _ = dataset

            print("shapes:")
            print("x shape: ", x.shape)
            print("y shape: ", y.shape)
            print("dydx shape: ", dydx.shape)
            if is_second_order:
                print("ddyddx shape: ", ddyddx.shape)


            xs = x[..., 0]
            ys = x[..., 1]


            fig_y = plot_3d_data(xs, ys, y, x1_label="x1", x2_label="x2", y_label="y", title=f"{name} target\ny")

            fig_x1 = plot_3d_data(xs, ys, dydx[:, 0], x1_label="x1", x2_label="x2", y_label="dydx1", title=f"{name} target\ndydx1")
            fig_x2 = plot_3d_data(xs, ys, dydx[:, 1], x1_label="x1", x2_label="x2", y_label="dydx1", title=f"{name} target\ndydx2")

            if is_second_order:
                fig_dx1dx1 = plot_3d_data(xs, ys, ddyddx[:, 0, 0], x1_label="x1", x2_label="x2", y_label="ddyddx11", title=f"{name} target\nddyddx11")
                fig_dx1dx2 = plot_3d_data(xs, ys, ddyddx[:, 0, 1], x1_label="x1", x2_label="x2", y_label="ddyddx12", title=f"{name} target\nddyddx12")
                fig_dx2dx1 = plot_3d_data(xs, ys, ddyddx[:, 1, 0], x1_label="x1", x2_label="x2", y_label="ddyddx21", title=f"{name} target\nddyddx21")
                fig_dx2dx2 = plot_3d_data(xs, ys, ddyddx[:, 1, 1], x1_label="x1", x2_label="x2", y_label="ddyddx22", title=f"{name} target\nddyddx22")


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
        