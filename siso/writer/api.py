from enum import Enum
from typing import Optional, Protocol, Sequence, TypeVar

from attrs import define
from typing_extensions import Self

from ..api import Field, Source, TimeStep, Endianness
from ..zone import Zone


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
    require_tesselated: bool = False
    require_instantaneous: bool = False


F = TypeVar("F", bound=Field)
T = TypeVar("T", bound=TimeStep)
Z = TypeVar("Z", bound=Zone)


class Writer(Protocol[F, T, Z]):
    def __enter__(self) -> Self:
        ...

    def __exit__(self, *args) -> None:
        ...

    @property
    def properties(self) -> WriterProperties:
        ...

    def configure(self, settings: WriterSettings):
        ...

    def consume(self, source: Source[F, T, Z], geometry: F, fields: Sequence[F]):
        ...
