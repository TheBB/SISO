from __future__ import annotations

from abc import ABC, abstractproperty
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator, List, Optional, Protocol, TypeVar, Union

import numpy as np
from typing_extensions import Self

from .util import FieldData
from .zone import Zone


class Endianness(Enum):
    Native = "native"
    Little = "little"
    Big = "big"

    def make_dtype(self, root: str) -> np.dtype:
        if self == Endianness.Native:
            return np.dtype(f"={root}")
        elif self == Endianness.Little:
            return np.dtype(f"<{root}")
        return np.dtype(f">{root}")

    def u4_type(self) -> np.dtype:
        return self.make_dtype("u4")

    def f4_type(self) -> np.dtype:
        return self.make_dtype("f4")


class Dimensionality(Enum):
    Volumetric = "volumetric"
    Planar = "planer"
    Extrude = "extrude"

    def out_is_volumetric(self) -> bool:
        return self != Dimensionality.Planar

    def in_allows_planar(self) -> bool:
        return self != Dimensionality.Volumetric


class Staggering(Enum):
    Outer = "outer"
    Inner = "inner"


@dataclass
class SplitFieldSpec:
    source_name: str
    new_name: str
    components: List[int]
    destroy: bool = True
    splittable: bool = False


@dataclass
class RecombineFieldSpec:
    source_names: List[str]
    new_name: str


@dataclass
class SourceProperties:
    instantaneous: bool
    globally_keyed: bool = False
    tesselated: bool = False
    single_zoned: bool = False

    split_fields: List[SplitFieldSpec] = field(default_factory=list)
    recombine_fields: List[RecombineFieldSpec] = field(default_factory=list)

    def update(self, **kwargs) -> SourceProperties:
        kwargs = {**self.__dict__, **kwargs}
        return SourceProperties(**kwargs)


class ScalarInterpretation(Enum):
    Generic = auto()
    Eigenmode = auto()

    def to_vector(self) -> VectorInterpretation:
        if self == ScalarInterpretation.Eigenmode:
            return VectorInterpretation.Eigenmode
        return VectorInterpretation.Generic


class VectorInterpretation(Enum):
    Generic = auto()
    Displacement = auto()
    Eigenmode = auto()
    Flow = auto()
    Geometry = auto()

    def join(self, other: VectorInterpretation) -> VectorInterpretation:
        if VectorInterpretation.Generic in (self, other):
            return VectorInterpretation.Generic
        assert self == other
        return self

    def to_scalar(self) -> ScalarInterpretation:
        if self == VectorInterpretation.Eigenmode:
            return ScalarInterpretation.Eigenmode
        return ScalarInterpretation.Generic


@dataclass
class Scalar:
    interpretation: ScalarInterpretation = ScalarInterpretation.Generic

    def slice(self) -> FieldType:
        return self

    def concat(self, other: FieldType) -> FieldType:
        if isinstance(other, Scalar):
            interpretation = self.interpretation.to_vector().join(other.interpretation.to_vector())
            return Vector(ncomps=2, interpretation=interpretation)
        interpretation = self.interpretation.to_vector().join(other.interpretation)
        return Vector(ncomps=other.ncomps + 1, interpretation=interpretation)


@dataclass
class Vector:
    ncomps: int
    interpretation: VectorInterpretation = VectorInterpretation.Generic

    def slice(self) -> FieldType:
        return Scalar(self.interpretation.to_scalar())

    def concat(self, other: FieldType) -> FieldType:
        if isinstance(other, Scalar):
            interpretation = self.interpretation.join(other.interpretation.to_vector())
            return Vector(ncomps=self.ncomps + 1, interpretation=interpretation)
        interpretation = self.interpretation.join(other.interpretation)
        return Vector(ncomps=self.ncomps + other.ncomps, interpretation=interpretation)

    def update(self, **kwargs) -> Vector:
        kwargs = {**self.__dict__, **kwargs}
        return Vector(**kwargs)


def Geometry(ncomps: int) -> Vector:
    return Vector(ncomps=ncomps, interpretation=VectorInterpretation.Geometry)


FieldType = Union[Scalar, Vector]


class Field:
    cellwise: bool
    splittable: bool
    name: str
    type: FieldType

    @property
    def is_scalar(self) -> bool:
        return isinstance(self.type, Scalar)

    @property
    def is_vector(self) -> bool:
        return isinstance(self.type, Vector)

    @property
    def is_geometry(self) -> bool:
        return isinstance(self.type, Vector) and self.type.interpretation == VectorInterpretation.Geometry

    @property
    def is_eigenmode(self) -> bool:
        return (
            isinstance(self.type, Scalar)
            and self.type.interpretation == ScalarInterpretation.Eigenmode
            or isinstance(self.type, Vector)
            and self.type.interpretation == VectorInterpretation.Eigenmode
        )

    @property
    def ncomps(self) -> int:
        if isinstance(self.type, Scalar):
            return 1
        return self.type.ncomps


class TimeStep(Protocol):
    @property
    def index(self) -> int:
        ...

    @property
    def time(self) -> Optional[float]:
        ...


@dataclass
class ReaderSettings:
    endianness: Endianness
    dimensionality: Dimensionality
    staggering: Staggering


Z = TypeVar("Z", bound=Zone)
F = TypeVar("F", bound=Field)
T = TypeVar("T", bound=TimeStep)


class Source(Protocol[F, T, Z]):
    @property
    def properties(self) -> SourceProperties:
        ...

    def configure(self, settings: ReaderSettings) -> None:
        ...

    def fields(self) -> Iterator[F]:
        ...

    def timesteps(self) -> Iterator[T]:
        ...

    def zones(self) -> Iterator[Z]:
        ...

    def topology(self, timestep: T, field: F, zone: Z) -> Topology:
        ...

    def field_data(self, timestep: T, field: F, zone: Z) -> FieldData:
        ...

    def use_geometry(self, geometry: F) -> None:
        ...

    def __enter__(self) -> Self:
        ...

    def __exit__(self, *args) -> None:
        ...


class CellType(Enum):
    Line = auto()
    Quadrilateral = auto()
    Hexahedron = auto()


class Topology(Protocol):
    @property
    def pardim(self) -> int:
        ...

    @property
    def num_nodes(self) -> int:
        ...

    @property
    def num_cells(self) -> int:
        ...

    def tesselator(self) -> Tesselator[Self]:
        ...


class DiscreteTopology(Topology, Protocol):
    @property
    def celltype(self) -> CellType:
        ...

    @property
    def cells(self) -> np.ndarray:
        ...


S = TypeVar("S", bound=Topology, contravariant=True)


class Tesselator(Protocol[S]):
    def tesselate_topology(self, topology: S) -> DiscreteTopology:
        ...

    def tesselate_field(self, topology: S, field: Field, field_data: FieldData) -> FieldData:
        ...
