from collections.abc import Callable, Generator
from typing_extensions import TypeAlias

from jaxtyping import Array, Float


#Data: TypeAlias = dict[str, Float[Array, "n_samples ..."]]

class DifferentialData():

    def __init__(self, order: int, x: Float[Array, "n*d"], y: Float[Array, "n"], dy: Float[Array, "n*d"], ddy: Float[Array, "n*d*d"] | None = None, dddy: Float[Array, "n*d*d*d"] | None = None):
        self.order = order
        self.x = x
        self.y = y
        self.dy = dy
        self.ddy = ddy
        self.dddy = dddy 

Scalar: TypeAlias = Float[Array, ""] # using 0-d arrays instead of float types to work with jax.vmap 

#DataGenerator: TypeAlias = Generator[Data, None, None]

Model: TypeAlias = Callable
