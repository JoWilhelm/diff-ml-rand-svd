import importlib.metadata

from diff_ml import losses, nn, smoothing, typing
from diff_ml.ad import hmp, hvp
from diff_ml.nn import train


__all__ = ["hmp", "hvp", "train", "losses", "nn", "smoothing", "typing"]

__version__ = importlib.metadata.version("diff-ml")
