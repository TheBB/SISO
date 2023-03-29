from typing import Iterator, Set

from .. import api
from ..api import B, F, S, T, Z
from .passthrough import PassthroughAll


class FieldFilter(PassthroughAll[B, F, S, T, Z]):
    filters: Set[str]

    def __init__(self, source: api.Source[B, F, S, T, Z], filters: Set[str]):
        super().__init__(source)
        self.filters = filters

    def fields(self, basis: B) -> Iterator[F]:
        for field in self.source.fields(basis):
            if field.name.casefold() in self.filters:
                yield field
