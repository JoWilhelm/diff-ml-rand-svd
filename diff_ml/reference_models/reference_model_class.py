from jaxtyping import Array, Float, PRNGKeyArray
from typing import Callable
from abc import ABC, abstractmethod

from diff_ml.typing import DifferentialData, Scalar


class ReferenceModel(ABC):


    def __init__(self, key_test: PRNGKeyArray, key_train: PRNGKeyArray, n_dims: int):
        self.key_test = key_test
        self.key_train = key_train
        self.n_dims = n_dims
        
    @abstractmethod
    def get_test_set(self, n_samples: int, order: int) -> DifferentialData: 
        """
        TODO
        """
        pass

    @abstractmethod
    def sample(self, key: PRNGKeyArray, n_samples: int, order: int = 1) -> DifferentialData: 
        """
        TODO
        """
        pass
    
    
    @abstractmethod
    def reference_fn(self) -> Callable[[Float[Array, "d"]], Scalar]:
        """
        TODO
        """
        pass
    
    
    @abstractmethod
    def visualize_data(self, dataset: DifferentialData, name: str):
        """
        TODO
        """
        pass
    

