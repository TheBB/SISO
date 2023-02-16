from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from .field import FieldType, FieldData
from .zone import Zone

from typing import (
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
)

from typing_extensions import Self


@dataclass
class SplitFieldSpec:
    source_name: str
    new_name: str
    components: List[int]
    destroy: bool


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


class Field(Protocol):
    @property
    def cellwise(self) -> bool:
        ...

    @property
    def name(self) -> str:
        ...

    @property
    def type(self) -> FieldType:
        ...

    @property
    def splittable(self) -> bool:
        ...

    @property
    def ncomps(self) -> int:
        ...


class TimeStep(Protocol):
    @property
    def index(self) -> int:
        ...

    @property
    def time(self) -> Optional[float]:
        ...


Z = TypeVar('Z', bound=Zone)
F = TypeVar('F', bound=Field)
T = TypeVar('T', bound=TimeStep)

class Source(Protocol[F, T, Z]):
    @property
    def properties(self) -> SourceProperties:
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

    def __enter__(self) -> Self:
        ...

    def __exit__(self, *args):
        ...


class CellType(Enum):
    Line = auto()
    Quadrilateral = auto()
    Hexahedron = auto()


class Topology(Protocol):
    @property
    def pardim(self) -> int:
        ...

    def tesselator(self) -> Tesselator[Self]:
        ...


class DiscreteTopology(Topology, Protocol):
    @property
    def celltype(self) -> CellType:
        ...

    @property
    def num_nodes(self) -> int:
        ...

    @property
    def num_cells(self) -> int:
        ...

    @property
    def cells(self) -> np.ndarray:
        ...


S = TypeVar('S', bound=Topology, contravariant=True)

class Tesselator(Protocol[S]):
    def tesselate_topology(self, topology: S) -> DiscreteTopology:
        ...

    def tesselate_field(self, topology: S, field: Field, field_data: FieldData) -> FieldData:
        ...
