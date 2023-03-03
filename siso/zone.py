from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Tuple

from attrs import define


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


@define
class Zone:
    shape: Shape
    coords: Coords
    local_key: Key
    global_key: Optional[int] = None
