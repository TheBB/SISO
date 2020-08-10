import importlib
from os.path import splitext

from .reader import Reader
from .ifem import IFEMReader, IFEMEigenReader
from .puregeometry import G2Reader, LRReader
from .simra import SIMRAReader
