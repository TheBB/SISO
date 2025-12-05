from __future__ import annotations

from typing import TYPE_CHECKING

from siso.api import Basis, Field, Step, Topology, Zone

from .passthrough import PassthroughAll

if TYPE_CHECKING:
    from collections.abc import Iterator

    from siso import api


class BasisFilter[B: Basis, F: Field, S: Step, T: Topology, Z: Zone](PassthroughAll[B, F, S, T, Z]):
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
