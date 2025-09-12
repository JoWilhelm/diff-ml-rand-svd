import importlib.metadata

from diff_ml import losses, nn, smoothing, typing, utils
#from diff_ml.ad import hmp, hvp
from diff_ml.nn import train


__all__ = ["train", "losses", "nn", "smoothing", "typing", "utils"]

__version__ = importlib.metadata.version("diff-ml")
