from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, cast

import erfa
import numpy as np
from attrs import define
from typing_extensions import Self

from . import api, util
from .util import FieldData


systems: util.Registry[api.CoordinateSystem] = util.Registry()
ellpsoids: util.Registry[Ellipsoid] = util.Registry()


def find_system(code: str) -> api.CoordinateSystem:
    name, *params = code.split(":")
    if name in systems:
        return systems[name].make(params)
    return Named.make((code,))


@systems.register
@define
class Generic(api.CoordinateSystem):
    name: ClassVar[str] = "Generic"

    @classmethod
    def make(cls, params: Sequence[str]) -> Generic:
        assert not params
        return cls()

    @classmethod
    def default(cls) -> Generic:
        return cls()

    @property
    def parameters(self) -> Tuple[str, ...]:
        return cast(Tuple[str], ())


@systems.register
@define
class Named(api.CoordinateSystem):
    name: ClassVar[str] = "Named"
    identifier: str

    @classmethod
    def make(cls, params: Sequence[str]) -> Named:
        (name,) = params
        return cls(name)

    @classmethod
    def default(cls) -> Named:
        assert False

    @property
    def parameters(self) -> Tuple[str, ...]:
        if self.identifier:
            return (self.identifier,)
        return cast(Tuple[str], ())

    def fits_system_name(self, code: str) -> bool:
        return code.casefold() == self.identifier.casefold()


@systems.register
@define
class Geodetic(api.CoordinateSystem):
    name: ClassVar[str] = "Geodetic"
    ellipsoid: Ellipsoid

    @classmethod
    def make(cls, params: Sequence[str]) -> Geodetic:
        assert len(params) < 2
        if params:
            return cls(ellpsoids[params[0]]())
        return cls(Wgs84())

    @classmethod
    def default(cls) -> Geodetic:
        return cls(Wgs84())

    @property
    def parameters(self) -> Tuple[str, ...]:
        return (self.ellipsoid.name,)


@systems.register
@define
class Utm(api.CoordinateSystem):
    name: ClassVar[str] = "UTM"
    zone_number: int
    zone_letter: str

    @classmethod
    def make(cls, params: Sequence[str]) -> Utm:
        (zone,) = params
        return cls(int(zone[:-1]), zone[-1].upper())

    @classmethod
    def default(cls) -> Utm:
        assert False

    @property
    def parameters(self) -> Tuple[str, ...]:
        return (str(self.zone_number), self.zone_letter)


@systems.register
class Geocentric(api.CoordinateSystem):
    name = "Geocentric"
    parameters = cast(Tuple[str], ())

    @classmethod
    def make(cls, params: Sequence[str]) -> Self:
        assert not params
        return cls()

    @classmethod
    def default(cls) -> Geocentric:
        return cls()


class Ellipsoid(ABC):
    name: ClassVar[str]

    @property
    @abstractmethod
    def semi_major_axis(self) -> float:
        ...

    @property
    @abstractmethod
    def flattening(self) -> float:
        ...


@ellpsoids.register
@define
class SphericalEarth(Ellipsoid):
    name = "Sphere"
    flattening: float = 0.0
    semi_major_axis: float = 6_371_008.8


class ErfaEllipsoid(Ellipsoid):
    erfa_code: ClassVar[int]

    @property
    def semi_major_axis(self) -> float:
        return erfa.eform(self.erfa_code)[0]

    @property
    def flattening(self) -> float:
        return erfa.eform(self.erfa_code)[1]


@ellpsoids.register
@define
class Wgs84(ErfaEllipsoid):
    erfa_code = 1
    name = "WGS84"


@ellpsoids.register
@define
class Grs80(ErfaEllipsoid):
    erfa_code = 2
    name = "GRS80"


@ellpsoids.register
@define
class Wgs72(ErfaEllipsoid):
    erfa_code = 3
    name = "WGS72"


T = TypeVar("T", bound=api.CoordinateSystem)
S = TypeVar("S", bound=api.CoordinateSystem)


CoordConverter = Callable[[T, S, FieldData], FieldData]
VectorConverter = Callable[[T, S, FieldData, FieldData], FieldData]
ConversionPath = List[api.CoordinateSystem]


NEIGHBORS: Dict[str, List[str]] = {}
COORD_CONVERTERS: Dict[Tuple[str, str], CoordConverter] = {}
VECTOR_CONVERTERS: Dict[Tuple[str, str], VectorConverter] = {}


def register_coords(
    src: Type[api.CoordinateSystem], tgt: Type[api.CoordinateSystem]
) -> Callable[[CoordConverter[T, S]], CoordConverter[T, S]]:
    def decorator(conv: CoordConverter) -> CoordConverter:
        NEIGHBORS.setdefault(src.name, []).append(tgt.name)
        COORD_CONVERTERS[(src.name, tgt.name)] = conv
        return conv

    return decorator


def register_vectors(src: str, tgt: str) -> Callable[[VectorConverter[T, S]], VectorConverter[T, S]]:
    def decorator(conv: VectorConverter) -> VectorConverter:
        VECTOR_CONVERTERS[(src, tgt)] = conv
        return conv

    return decorator


def conversion_path(src: api.CoordinateSystem, tgt: api.CoordinateSystem) -> Optional[ConversionPath]:
    if src == tgt:
        return []
    if isinstance(src, (Generic, Named)) and isinstance(tgt, Generic):
        return []

    visited: Dict[str, str] = {}
    queue: deque[str] = deque((src.name,))

    def construct_backpath() -> ConversionPath:
        path = [tgt]
        name = visited[tgt.name]
        while name != src.name:
            path.append(systems[name].default())
            name = visited[name]
        path.append(src)
        return path[::-1]

    while queue:
        system = queue.popleft()
        for neighbor in NEIGHBORS.get(system, []):
            if neighbor in visited or neighbor == src.name:
                continue
            visited[neighbor] = system
            if neighbor == tgt.name:
                return construct_backpath()
            queue.append(neighbor)

    return None


def optimal_system(
    systems: Sequence[api.CoordinateSystem], target: api.CoordinateSystem
) -> Optional[Tuple[int, ConversionPath]]:
    optimal: Optional[Tuple[int, ConversionPath]] = None

    for i, system in enumerate(systems):
        new_path = conversion_path(system, target)
        if new_path is None:
            continue
        if optimal is None:
            optimal = i, new_path
        _, prev_path = optimal
        if len(new_path) < len(prev_path):
            optimal = i, new_path

    return optimal


def convert_coords(
    src: api.CoordinateSystem,
    tgt: api.CoordinateSystem,
    data: FieldData,
) -> FieldData:
    return COORD_CONVERTERS[(src.name, tgt.name)](src, tgt, data)


def convert_vectors(
    src: api.CoordinateSystem,
    tgt: api.CoordinateSystem,
    data: FieldData,
    coords: FieldData,
) -> FieldData:
    return VECTOR_CONVERTERS[(src.name, tgt.name)](src, tgt, data, coords)


@register_coords(Geodetic, Geocentric)
def _(src: Geodetic, tgt: Geocentric, data: FieldData) -> FieldData:
    lon, lat, height = data.components
    return FieldData(
        erfa.gd2gce(
            src.ellipsoid.semi_major_axis,
            src.ellipsoid.flattening,
            np.deg2rad(lon),
            np.deg2rad(lat),
            height,
        )
    )


@register_vectors("Geodetic", "Geocentric")
def _(src: Geodetic, tgt: Geocentric, data: FieldData, coords: FieldData) -> FieldData:
    return data.spherical_to_cartesian_vector_field(coords)
