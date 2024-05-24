from collections.abc import Callable, Generator
from typing import Optional
from typing_extensions import TypeAlias

from jaxtyping import Array, Float, PRNGKeyArray


Data: TypeAlias = dict[str, Float[Array, "n_samples ..."]]

DataGenerator: TypeAlias = Generator[Data, None, None]

# Model: TypeAlias = Callable[[Array, Optional[PRNGKeyArray]], Array]
Model: TypeAlias = Callable
