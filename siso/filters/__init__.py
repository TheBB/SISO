from .decompose import Decompose, Split
from .force_unstructured import ForceUnstructured
from .keyzones import KeyZones
from .recombine import Recombine
from .tesselate import Tesselate
from .zonemerge import ZoneMerge

from ..api import Source

from typing import Callable


Filter = Callable[[Source], Source]
