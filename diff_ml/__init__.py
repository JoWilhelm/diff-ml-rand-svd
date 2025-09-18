import importlib.metadata
from diff_ml import losses, nn, typing, utils
from diff_ml.nn import train


__all__ = ["train", "losses", "nn", "typing", "utils"]

__version__ = importlib.metadata.version("diff-ml")
