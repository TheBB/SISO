from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import (
    ClassVar,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

import numpy as np
from attrs import Factory, asdict, define
from numpy import floating, integer
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


@define
class SplitFieldSpec:
    source_name: str
    new_name: str
    components: List[int]
    destroy: bool = True
    splittable: bool = False


@define
class RecombineFieldSpec:
    source_names: List[str]
    new_name: str


class StepInterpretation(Enum):
    Time = auto()
    Eigenmode = auto()
    EigenFrequency = auto()

    @property
    def is_time(self):
        return self == StepInterpretation.Time

    @property
    def is_eigen(self):
        return self != StepInterpretation.Time

    def __str__(self) -> str:
        return {
            StepInterpretation.Time: "Time",
            StepInterpretation.Eigenmode: "Eigenvalue",
            StepInterpretation.EigenFrequency: "Frequency",
        }[self]


@define
class SourceProperties:
    instantaneous: bool
    globally_keyed: bool = False
    tesselated: bool = False
    single_zoned: bool = False
    step_interpretation: StepInterpretation = StepInterpretation.Time

    split_fields: List[SplitFieldSpec] = Factory(list)
    recombine_fields: List[RecombineFieldSpec] = Factory(list)

    def update(self, **kwargs) -> SourceProperties:
        kwargs = {**asdict(self, recurse=False), **kwargs}
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

    def join(self, other: VectorInterpretation) -> VectorInterpretation:
        if VectorInterpretation.Generic in (self, other):
            return VectorInterpretation.Generic
        assert self == other
        return self

    def to_scalar(self) -> ScalarInterpretation:
        if self == VectorInterpretation.Eigenmode:
            return ScalarInterpretation.Eigenmode
        return ScalarInterpretation.Generic


@define
class Scalar:
    interpretation: ScalarInterpretation = ScalarInterpretation.Generic

    def slice(self) -> FieldType:
        return self

    def concat(self, other: FieldType) -> FieldType:
        if isinstance(other, Scalar):
            interpretation = self.interpretation.to_vector().join(other.interpretation.to_vector())
            return Vector(ncomps=2, interpretation=interpretation)
        assert isinstance(other, Vector)
        interpretation = self.interpretation.to_vector().join(other.interpretation)
        return Vector(ncomps=other.ncomps + 1, interpretation=interpretation)


@define
class Vector:
    ncomps: int
    interpretation: VectorInterpretation = VectorInterpretation.Generic

    def slice(self) -> FieldType:
        return Scalar(self.interpretation.to_scalar())

    def concat(self, other: FieldType) -> FieldType:
        if isinstance(other, Scalar):
            interpretation = self.interpretation.join(other.interpretation.to_vector())
            return Vector(ncomps=self.ncomps + 1, interpretation=interpretation)
        assert isinstance(other, Vector)
        interpretation = self.interpretation.join(other.interpretation)
        return Vector(ncomps=self.ncomps + other.ncomps, interpretation=interpretation)

    def update(self, **kwargs) -> Vector:
        kwargs = {**asdict(self, recurse=False), **kwargs}
        return Vector(**kwargs)


@define
class Geometry:
    ncomps: int
    coords: CoordinateSystem

    def slice(self) -> FieldType:
        assert False

    def concat(self, other: FieldType) -> FieldType:
        assert False

    def fits_system_name(self, name: Optional[str]) -> bool:
        if name is None:
            return True
        return self.coords.fits_system_name(name)


class CoordinateSystem(ABC):
    name: ClassVar[str]

    @classmethod
    @abstractmethod
    def make(cls, params: Sequence[str]) -> Self:
        ...

    @classmethod
    @abstractmethod
    def default(cls) -> Self:
        ...

    @property
    @abstractmethod
    def parameters(self) -> Tuple[str, ...]:
        ...

    def fits_system_name(self, code: str) -> bool:
        return code.casefold() == self.name.casefold()

    def __str__(self) -> str:
        params = ", ".join(p for p in self.parameters)
        return f"{self.name}({params})"


FieldType = Union[Scalar, Vector, Geometry]


class Field(ABC):
    @property
    @abstractmethod
    def cellwise(self) -> bool:
        ...

    @property
    @abstractmethod
    def splittable(self) -> bool:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def type(self) -> FieldType:
        ...

    @property
    def is_scalar(self) -> bool:
        return isinstance(self.type, Scalar)

    @property
    def is_vector(self) -> bool:
        return isinstance(self.type, Vector)

    @property
    def is_geometry(self) -> bool:
        return isinstance(self.type, Geometry)

    @property
    def is_eigenmode(self) -> bool:
        return (
            isinstance(self.type, Scalar)
            and self.type.interpretation == ScalarInterpretation.Eigenmode
            or isinstance(self.type, Vector)
            and self.type.interpretation == VectorInterpretation.Eigenmode
        )

    @property
    def is_displacement(self) -> bool:
        return isinstance(self.type, Vector) and self.type.interpretation == VectorInterpretation.Displacement

    @property
    def coords(self) -> CoordinateSystem:
        return cast(Geometry, self.type).coords

    def fits_system_name(self, code: str) -> bool:
        return isinstance(self.type, Geometry) and self.type.fits_system_name(code)

    @property
    def ncomps(self) -> int:
        if isinstance(self.type, Scalar):
            return 1
        return self.type.ncomps


class Step(Protocol):
    @property
    def index(self) -> int:
        ...

    @property
    def value(self) -> Optional[float]:
        ...


@define
class ReaderSettings:
    endianness: Endianness
    dimensionality: Dimensionality
    staggering: Staggering
    periodic: bool


Z = TypeVar("Z", bound=Zone)
F = TypeVar("F", bound=Field)
S = TypeVar("S", bound=Step)


class Source(ABC, Generic[F, S, Z]):
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        return

    @property
    @abstractmethod
    def properties(self) -> SourceProperties:
        ...

    def configure(self, settings: ReaderSettings) -> None:
        return

    def use_geometry(self, geometry: F) -> None:
        return

    @abstractmethod
    def fields(self) -> Iterator[F]:
        ...

    @abstractmethod
    def steps(self) -> Iterator[S]:
        ...

    @abstractmethod
    def zones(self) -> Iterator[Z]:
        ...

    @abstractmethod
    def topology(self, timestep: S, field: F, zone: Z) -> Topology:
        ...

    @abstractmethod
    def field_data(self, timestep: S, field: F, zone: Z) -> FieldData[floating]:
        ...

    def field_updates(self, timestep: S, field: F) -> bool:
        return True


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

    def tesselator(self, nvis: int = 1) -> Tesselator[Self]:
        ...


@runtime_checkable
class DiscreteTopology(Topology, Protocol):
    @property
    def celltype(self) -> CellType:
        ...

    @property
    def cells(self) -> FieldData[integer]:
        ...


T = TypeVar("T", bound=Topology, contravariant=True)


class Tesselator(Protocol[T]):
    def tesselate_topology(self, topology: T) -> DiscreteTopology:
        ...

    def tesselate_field(
        self, topology: T, field: Field, field_data: FieldData[floating]
    ) -> FieldData[floating]:
        ...
