from jaxtyping import Array, Float, PRNGKeyArray
from typing import Callable, Any
from diff_ml.typing import DifferentialData, Scalar
from abc import ABC, abstractmethod


class ReferenceModel(ABC):

    key_test: PRNGKeyArray
    key_train: PRNGKeyArray
    n_dims: int  # input dimension


    def __init__(self, key_test: PRNGKeyArray, key_train: PRNGKeyArray, n_dims: int):
        self.key_test = key_test
        self.key_train = key_train
        self.n_dims = n_dims
        
    @abstractmethod
    def get_test_set(self, n_samples: int, order: int) -> DifferentialData: 
        """
        Get a test set of n_samples from the reference model's input distribution, returning the corresponding outputs and derivatives up to the specified order.
        Use self.key_test internally.
        """
        pass

    @abstractmethod
    def sample(self, key: PRNGKeyArray, n_samples: int, order: int) -> DifferentialData: 
        """
        Sample n_samples from the reference model's input distribution and return the corresponding outputs and derivatives up to the specified order.
        """
        pass
    
    
    @abstractmethod
    # returns a function form n_dims to scalar
    def reference_fn(self) -> Callable[[Float[Array, "n_dims"], PRNGKeyArray], Scalar]:
        """
        Must accept xs in the same space as returned by sample(). I.e. if sample() returns normalized data, the returned function must accept normalized data. 
        They key argument is optional, depending on whether the reference model is stochastic or not.
        """
        pass
    
    
    @abstractmethod
    def visualize_data(self, dataset: DifferentialData, name: str) -> Any:
        """
        Plot or visualize the dataset in some way, depending on the model.
        """
        pass
    

