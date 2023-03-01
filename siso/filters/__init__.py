from typing import Callable, TypeVar

from .. import api
from ..zone import Zone
from .decompose import Decompose, Split
from .eigendisp import EigenDisp
from .force_unstructured import ForceUnstructured
from .keyzones import KeyZones
from .recombine import Recombine
from .strict import Strict
from .tesselate import Tesselate
from .zonemerge import ZoneMerge


InZ = TypeVar("InZ", bound=Zone)
InF = TypeVar("InF", bound=api.Field)
InT = TypeVar("InT", bound=api.TimeStep)
OutZ = TypeVar("OutZ", bound=Zone)
OutF = TypeVar("OutF", bound=api.Field)
OutT = TypeVar("OutT", bound=api.TimeStep)

Filter = Callable[[api.Source[InF, InT, InZ]], api.Source[InF, InT, InZ]]
