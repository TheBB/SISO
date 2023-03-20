from typing import Iterator, Set, TypeVar

from .. import api
from .passthrough import PassthroughAll


B = TypeVar("B", bound=api.Basis)
F = TypeVar("F", bound=api.Field)
S = TypeVar("S", bound=api.Step)
Z = TypeVar("Z", bound=api.Zone)


class BasisFilter(PassthroughAll[B, F, S, Z]):
    filters: Set[str]

    def __init__(self, source: api.Source[B, F, S, Z], filters: Set[str]):
        super().__init__(source)
        self.filters = filters

    def bases(self) -> Iterator[B]:
        for basis in self.source.bases():
            if basis.name.casefold() in self.filters:
                yield basis
