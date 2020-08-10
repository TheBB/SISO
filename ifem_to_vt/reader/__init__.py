import importlib
from os.path import splitext

from .reader import Reader
from .hdf5 import IFEMReader, IFEMEigenReader
from .g2 import G2Reader
from .lr import LRReader
from .res import SIMRAReader
