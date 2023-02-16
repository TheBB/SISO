from .keyzones import KeyZones
from .tesselate import Tesselate
from .zonemerge import ZoneMerge
from .force_unstructured import ForceUnstructured

from ..api import Source

from typing import Callable


Filter = Callable[[Source], Source]
