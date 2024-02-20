from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional, Protocol

from attrs import define
from typing_extensions import Self

from siso.api import B, Endianness, F, S, Source, T, Z

if TYPE_CHECKING:
    from types import TracebackType


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


class Writer(Protocol):
    def __enter__(self) -> Self:
        ...

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        ...

    @property
    def properties(self) -> WriterProperties:
        ...

    def configure(self, settings: WriterSettings) -> None:
        ...

    def consume(self, source: Source[B, F, S, T, Z], geometry: F) -> None:
        ...
