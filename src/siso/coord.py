from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING, ClassVar, Protocol, Self, cast

import erfa
import numpy as np
from attrs import define

from siso.types import Floatd

from . import api, util
from .util import FieldData, coord

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from siso.types import Floatd
    from siso.util.field_data import FloatFieldData

systems: util.Registry[type[api.CoordinateSystem]] = util.Registry()
ellpsoids: util.Registry[type[Ellipsoid]] = util.Registry()


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
        if params:
            raise api.BadInput("Generic coordinate system does not accept parameters")
        return cls()

    @classmethod
    def default(cls) -> Generic:
        return cls()

    @property
    def parameters(self) -> tuple[str, ...]:
        return cast("tuple[str]", ())


@systems.register
@define
class Named(api.CoordinateSystem):
    name: ClassVar[str] = "Named"
    identifier: str

    @classmethod
    def make(cls, params: Sequence[str]) -> Named:
        if len(params) != 1:
            raise api.BadInput("Named coordinate system requires one parameter")
        (name,) = params
        return cls(name)

    @classmethod
    def default(cls) -> Named:
        raise api.Unsupported("There is no default named coordinate system")

    @property
    def parameters(self) -> tuple[str, ...]:
        if self.identifier:
            return (self.identifier,)
        return cast("tuple[str]", ())

    def fits_system_name(self, code: str) -> bool:
        return code.casefold() == self.identifier.casefold()


@systems.register
@define
class Geodetic(api.CoordinateSystem):
    name: ClassVar[str] = "Geodetic"
    ellipsoid: Ellipsoid

    @classmethod
    def make(cls, params: Sequence[str]) -> Geodetic:
        if len(params) >= 2:
            raise api.BadInput("Geodetic coordinate system only accepts one optional parameter")
        if params:
            return cls(ellpsoids[params[0]]())
        return cls(Wgs84())

    @classmethod
    def default(cls) -> Geodetic:
        return cls(Wgs84())

    @property
    def parameters(self) -> tuple[str, ...]:
        return (self.ellipsoid.name,)

    @property
    def semi_major_axis(self) -> float:
        return self.ellipsoid.semi_major_axis

    @property
    def flattening(self) -> float:
        return self.ellipsoid.flattening


@systems.register
@define
class Utm(api.CoordinateSystem):
    name: ClassVar[str] = "UTM"
    zone_number: int
    northern: bool

    @classmethod
    def make(cls, params: Sequence[str]) -> Utm:
        (zone,) = params
        try:
            i = next(i for i in range(len(zone)) if not zone[i].isnumeric())
        except StopIteration:
            raise ValueError(zone)
        zone_number = int(zone[:i])
        lat_band = zone[i:]
        northern = lat_band.casefold() >= "N" if len(lat_band) == 1 else lat_band[0].casefold() != "S"
        return cls(zone_number, northern)

    @classmethod
    def default(cls) -> Utm:
        raise api.Unsupported("There is no default UTM coordinate system")

    @property
    def parameters(self) -> tuple[str, ...]:
        return (str(self.zone_number), "north" if self.northern else "south")


@systems.register
class Geocentric(api.CoordinateSystem):
    name: ClassVar[str] = "Geocentric"

    @property
    def parameters(self) -> tuple[str, ...]:
        return ()

    @classmethod
    def make(cls, params: Sequence[str]) -> Self:
        if params:
            raise api.BadInput("Geocentric coordinate system does not accept parameters")
        return cls()

    @classmethod
    def default(cls) -> Geocentric:
        return cls()


class Ellipsoid(ABC):
    name: ClassVar[str]

    @property
    @abstractmethod
    def semi_major_axis(self) -> float: ...

    @property
    @abstractmethod
    def flattening(self) -> float: ...


@ellpsoids.register
@define
class SphericalEarth(Ellipsoid):
    name: ClassVar[str] = "Sphere"
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
    name: ClassVar[str] = "WGS84"


@ellpsoids.register
@define
class Grs80(ErfaEllipsoid):
    erfa_code = 2
    name: ClassVar[str] = "GRS80"


@ellpsoids.register
@define
class Wgs72(ErfaEllipsoid):
    erfa_code = 3
    name: ClassVar[str] = "WGS72"


class CoordConverter[S: api.CoordinateSystem, T: api.CoordinateSystem](Protocol):
    def __call__[D: Floatd](self, src: S, tgt: T, data: FloatFieldData) -> FloatFieldData: ...


class VectorConverter[S: api.CoordinateSystem, T: api.CoordinateSystem](Protocol):
    def __call__(self, src: S, tgt: T, data: FloatFieldData, coords: FloatFieldData) -> FloatFieldData: ...


class CoordConverterRegistrator(Protocol):
    def __call__[S: api.CoordinateSystem, T: api.CoordinateSystem](
        self, conv: CoordConverter[S, T]
    ) -> CoordConverter[S, T]: ...


class VectorConverterRegistrator(Protocol):
    def __call__[S: api.CoordinateSystem, T: api.CoordinateSystem](
        self, conv: VectorConverter[S, T]
    ) -> VectorConverter[S, T]: ...


type ConversionPath = list[api.CoordinateSystem]


NEIGHBORS: dict[str, list[str]] = {}
COORD_CONVERTERS: dict[tuple[str, str], CoordConverter] = {}
VECTOR_CONVERTERS: dict[tuple[str, str], VectorConverter] = {}


def register_coords[S: api.CoordinateSystem, T: api.CoordinateSystem](
    src: type[S], tgt: type[T]
) -> Callable[[CoordConverter[S, T]], CoordConverter[S, T]]:
    def decorator(conv: CoordConverter) -> CoordConverter:
        NEIGHBORS.setdefault(src.name, []).append(tgt.name)
        COORD_CONVERTERS[(src.name, tgt.name)] = conv
        return conv

    return decorator


def register_vectors(
    src: type[api.CoordinateSystem], tgt: type[api.CoordinateSystem]
) -> VectorConverterRegistrator:
    def decorator(conv: VectorConverter) -> VectorConverter:
        VECTOR_CONVERTERS[(src.name, tgt.name)] = conv
        return conv

    return decorator


def conversion_path(src: api.CoordinateSystem, tgt: api.CoordinateSystem) -> ConversionPath | None:
    if src == tgt:
        return []
    if isinstance(src, Generic | Named) and isinstance(tgt, Generic):
        return []

    visited: dict[str, str] = {}
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
) -> tuple[int, ConversionPath] | None:
    optimal: tuple[int, ConversionPath] | None = None

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


def convert_coords[D: Floatd](
    src: api.CoordinateSystem,
    tgt: api.CoordinateSystem,
    data: FloatFieldData,
) -> FloatFieldData:
    return COORD_CONVERTERS[(src.name, tgt.name)](src, tgt, data)


def convert_vectors(
    src: api.CoordinateSystem,
    tgt: api.CoordinateSystem,
    data: FloatFieldData,
    coords: FloatFieldData,
) -> FloatFieldData:
    return VECTOR_CONVERTERS[(src.name, tgt.name)](src, tgt, data, coords)


@register_coords(Geodetic, Geocentric)
def _[D: Floatd](src: Geodetic, tgt: Geocentric, data: FloatFieldData) -> FloatFieldData:
    lon, lat, height = data.comps
    return FieldData(
        erfa.gd2gce(
            src.ellipsoid.semi_major_axis,
            src.ellipsoid.flattening,
            np.deg2rad(lon),
            np.deg2rad(lat),
            height,
        )
    )


@register_vectors(Geodetic, Geocentric)
def _[D: Floatd](
    src: Geodetic, tgt: Geocentric, data: FloatFieldData, coords: FloatFieldData
) -> FloatFieldData:
    return data.spherical_to_cartesian_vector_field(coords)


@register_coords(Geodetic, Utm)
def _(src: Geodetic, tgt: Utm, data: FloatFieldData) -> FloatFieldData:
    lon, lat, *rest = data.comps
    converter = coord.UtmConverter(src.semi_major_axis, src.flattening, tgt.zone_number, tgt.northern)
    x, y = converter.to_utm(lon, lat)
    return FieldData.join_comps(x, y, *rest)


@register_vectors(Geodetic, Utm)
def _(src: Geodetic, tgt: Utm, data: FloatFieldData, coords: FloatFieldData) -> FloatFieldData:
    lon, lat, *_ = coords.comps
    in_x, in_y, *rest = data.comps
    converter = coord.UtmConverter(src.semi_major_axis, src.flattening, tgt.zone_number, tgt.northern)
    out_x, out_y = converter.to_utm_vf(lon, lat, in_x, in_y)
    return FieldData.join_comps(out_x, out_y, *rest)


@register_coords(Utm, Geodetic)
def _(src: Utm, tgt: Geodetic, data: FloatFieldData) -> FloatFieldData:
    x, y, *rest = data.comps
    converter = coord.UtmConverter(tgt.semi_major_axis, tgt.flattening, src.zone_number, src.northern)
    lon, lat = converter.to_lonlat(x, y)
    return FieldData.join_comps(lon, lat, *rest)


@register_vectors(Utm, Geodetic)
def _(src: Utm, tgt: Geodetic, data: FloatFieldData, coords: FloatFieldData) -> FloatFieldData:
    x, y, *_ = coords.comps
    in_x, in_y, *rest = data.comps
    converter = coord.UtmConverter(tgt.semi_major_axis, tgt.flattening, src.zone_number, src.northern)
    out_x, out_y = converter.to_lonlat_vf(x, y, in_x, in_y)
    return FieldData.join_comps(out_x, out_y, *rest)
