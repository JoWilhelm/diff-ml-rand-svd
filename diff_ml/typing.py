from collections.abc import Callable, Generator
from typing_extensions import TypeAlias

from jaxtyping import Array, Float


Data: TypeAlias = dict[str, Float[Array, "n_samples ..."]]

DataGenerator: TypeAlias = Generator[Data, None, None]

Model: TypeAlias = Callable
