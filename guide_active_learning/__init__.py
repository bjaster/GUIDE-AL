try:
    from ._version import version as __version__
except ImportError:
    __version__ = "UNKNOWN"

from .core import *

from . import ActiveLearning, GUIDE