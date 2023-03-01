from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Generic, Optional, Tuple, TypeVar


Point = Tuple[float, ...]
Coords = Tuple[Point, ...]
Key = str


class Shape(Enum):
    # 1D
    Line = auto()

    # 2D
    Triangle = auto()
    Quatrilateral = auto()

    # 3D
    Hexahedron = auto()

    # Misc
    Shapeless = auto()


@dataclass
class Zone:
    shape: Shape
    coords: Coords
    local_key: Key
    global_key: Optional[int] = None
