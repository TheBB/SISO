from __future__ import annotations

from typing import TYPE_CHECKING

from siso.api import B, F, S, T, Z

from .passthrough import PassthroughAll

if TYPE_CHECKING:
    from collections.abc import Iterator

    from siso import api


class FieldFilter(PassthroughAll[B, F, S, T, Z]):
    """Filter that removes fields that don't match the set of allowed names."""

    allowed_names: set[str]

    def __init__(self, source: api.Source[B, F, S, T, Z], allowed_names: set[str]):
        super().__init__(source)
        self.allowed_names = allowed_names

    def fields(self, basis: B) -> Iterator[F]:
        for field in self.source.fields(basis):
            if field.name.casefold() in self.allowed_names:
                yield field
