from dataclasses import dataclass
from enum import Enum, auto

from ..api import Source, TimeStep
from ..field import Field
from ..zone import Zone

from typing import Protocol, Sequence, TypeVar, Optional
from typing_extensions import Self


class OutputMode(Enum):
    Binary = 'binary'
    Ascii = 'ascii'
    Appended = 'appended'


@dataclass
class WriterSettings:
    output_mode: Optional[OutputMode] = None


@dataclass
class WriterProperties:
    require_single_zone: bool = False
    require_tesselated: bool = False


F = TypeVar('F', bound=Field)
T = TypeVar('T', bound=TimeStep)
Z = TypeVar('Z', bound=Zone)

class Writer(Protocol):
    def __enter__(self) -> Self:
        ...

    def __exit__(self, *args):
        ...

    @property
    def properties(self) -> WriterProperties:
        ...

    def configure(self, settings: WriterSettings):
        ...

    def consume(self, source: Source[F, T, Z], geometry: F, fields: Sequence[F]):
        ...
