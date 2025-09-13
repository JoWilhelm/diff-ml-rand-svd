from jaxtyping import Array, Float, PRNGKeyArray
from typing import Callable
from abc import ABC, abstractmethod

from diff_ml.typing import Data


class ReferenceModel(ABC):


    def __init__(self, key: PRNGKeyArray, n_dims: int, unflattened_shape: tuple):
        self.key = key
        self.n_dims = n_dims
        self.unflattened_shape = unflattened_shape

    @abstractmethod
    def get_testset(self, n_samples: int) -> Data: 
        """
        TODO
        """
        pass

    @abstractmethod
    def sample(self, key: PRNGKeyArray, n_samples: int, higher_order: bool = False) -> Data: 
        """
        TODO
        """
        pass
    
    
    @abstractmethod
    def reference_fn(self) -> Callable[[Float[Array, "d"]], float]:
        """
        TODO
        """
        pass
    


