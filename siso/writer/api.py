from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Protocol, Sequence, TypeVar

from typing_extensions import Self

from ..api import Field, Source, TimeStep
from ..zone import Zone


class OutputMode(Enum):
    Binary = "binary"
    Ascii = "ascii"
    Appended = "appended"


@dataclass
class WriterSettings:
    output_mode: Optional[OutputMode] = None


@dataclass
class WriterProperties:
    require_single_zone: bool = False
    require_tesselated: bool = False


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
