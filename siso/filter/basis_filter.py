from collections.abc import Iterator

from siso import api
from siso.api import B, F, S, T, Z

from .passthrough import PassthroughAll


class BasisFilter(PassthroughAll[B, F, S, T, Z]):
    """Source filter that removes some bases.

    Parameters:
    - source: data source to draw from.
    - filters: set of basis names to allow through.
    """

    filters: set[str]

    def __init__(self, source: api.Source[B, F, S, T, Z], filters: set[str]):
        super().__init__(source)
        self.filters = filters

    def bases(self) -> Iterator[B]:
        for basis in self.source.bases():
            if basis.name.casefold() in self.filters:
                yield basis
