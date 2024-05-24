import importlib.metadata

from diff_ml import losses, nn, typing
from diff_ml.ad import hmp, hvp
from diff_ml.nn import train


__all__ = ["hmp", "hvp", "train", "losses", "nn", "typing"]

__version__ = importlib.metadata.version("diff-ml")
