from enum import Enum
from typing import Optional, Protocol, TypeVar

from attrs import define
from typing_extensions import Self

from ..api import Basis, Endianness, Field, Source, Step, Topology, Zone


class OutputMode(Enum):
    Binary = "binary"
    Ascii = "ascii"
    Appended = "appended"


@define
class WriterSettings:
    output_mode: Optional[OutputMode] = None
    endianness: Endianness = Endianness.Native


@define
class WriterProperties:
    require_single_zone: bool = False
    require_discrete_topology: bool = False
    require_single_basis: bool = False
    require_instantaneous: bool = False


B = TypeVar("B", bound=Basis)
F = TypeVar("F", bound=Field)
S = TypeVar("S", bound=Step)
T = TypeVar("T", bound=Topology)
Z = TypeVar("Z", bound=Zone)


class Writer(Protocol[B, F, S, T, Z]):
    def __enter__(self) -> Self:
        ...

    def __exit__(self, *args) -> None:
        ...

    @property
    def properties(self) -> WriterProperties:
        ...

    def configure(self, settings: WriterSettings):
        ...

    def consume(self, source: Source[B, F, S, T, Z], geometry: F):
        ...
