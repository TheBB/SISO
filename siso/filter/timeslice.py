from itertools import islice
from typing import Iterator, Optional, Tuple, TypeVar

from .. import api
from .passthrough import Passthrough


F = TypeVar("F", bound=api.Field)
Z = TypeVar("Z", bound=api.Zone)
T = TypeVar("T", bound=api.TimeStep)


class TimeSlice(Passthrough[F, T, Z, F, T, Z]):
    arguments: Tuple[Optional[int]]

    def __init__(self, source: api.Source[F, T, Z], arguments: Tuple[Optional[int]]):
        super().__init__(source)
        self.arguments = arguments

    def timesteps(self) -> Iterator[T]:
        return islice(self.source.timesteps(), *self.arguments)
