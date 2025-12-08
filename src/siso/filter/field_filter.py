from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from siso.api import Basis, Field, Step, Topology, Zone, impl_fields

from .passthrough import PassthroughAll

if TYPE_CHECKING:
    from collections.abc import Iterator

    from siso import api


class FieldFilter[B: Basis, F: Field, S: Step, T: Topology, Z: Zone](PassthroughAll[B, F, S, T, Z]):
    """Filter that removes fields that don't match the set of allowed names."""

    allowed_names: set[str] | None
    disallowed_names: set[str] | None | Literal["all"]

    def __init__(
        self,
        source: api.Source[B, F, S, T, Z],
        allowed_names: set[str] | None,
        disallowed_names: set[str] | None | Literal["all"],
    ):
        super().__init__(source)
        self.allowed_names = allowed_names
        self.disallowed_names = disallowed_names

    @impl_fields
    def fields(self, basis: B) -> Iterator[F]:
        for field in self.source.fields(basis):
            if self.disallowed_names == "all":
                continue
            if self.disallowed_names is not None and field.name.casefold() in self.disallowed_names:
                continue
            if self.allowed_names is not None and field.name.casefold() not in self.allowed_names:
                continue
            yield field
