from typing import Iterator, TypeVar

from .. import api
from .passthrough import Passthrough


F = TypeVar("F", bound=api.Field)
Z = TypeVar("Z", bound=api.Zone)
T = TypeVar("T", bound=api.TimeStep)


class LastTime(Passthrough[F, T, Z, F, T, Z]):
    def timesteps(self) -> Iterator[T]:
        timesteps = self.source.timesteps()
        timestep = next(timesteps)
        for timestep in timesteps:
            pass
        yield timestep
