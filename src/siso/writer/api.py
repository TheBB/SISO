from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol, Self

from attrs import define

from siso.api import Basis, Endianness, Field, Source, Step, Topology, Zone

if TYPE_CHECKING:
    from types import TracebackType


class OutputMode(Enum):
    Binary = "binary"
    Ascii = "ascii"
    Appended = "appended"


@define
class WriterSettings:
    output_mode: OutputMode | None = None
    endianness: Endianness = Endianness.Native


@define
class WriterProperties:
    require_single_zone: bool = False
    require_discrete_topology: bool = False
    require_single_basis: bool = False
    require_instantaneous: bool = False


class Writer(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None: ...

    @property
    def properties(self) -> WriterProperties: ...

    def configure(self, settings: WriterSettings) -> None: ...

    def consume[B: Basis, F: Field, S: Step, T: Topology, Z: Zone](
        self, source: Source[B, F, S, T, Z], geometry: F
    ) -> None: ...
