from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

from attrs import define

from siso.api import B, F, S, T, Z
from siso.util import FieldData

from .passthrough import PassthroughBSTZ, WrappedField

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy import floating

    from siso import api


@define
class RecombinedField(WrappedField[F]):
    """Class for a 'recombined' field: a field that combines components from
    multiple sources into one.

    If there is only one source, this acts as a faithful reprodection of the
    source field.
    """

    sources: list[F]
    name: str

    def __post_init__(self) -> None:
        # Assert that the combined fields are compatible
        assert all(src.cellwise == self.sources[0].cellwise for src in self.sources)
        assert all(src.type == self.sources[0].type for src in self.sources)

    @property
    def wrapped_field(self) -> F:
        return self.sources[0]

    @property
    def type(self) -> api.FieldType:
        return reduce(lambda x, y: x.join(y), (s.type for s in self.sources))

    @property
    def splittable(self) -> bool:
        # Don't split a recombined field (unless it's a faithful reproduction).
        if len(self.sources) == 1:
            return self.sources[0].splittable
        return False


class Recombine(PassthroughBSTZ[B, S, T, Z, F, RecombinedField[F]]):
    """Filter that recombines fields as indicated by a list of
    `RecombineFieldSpec` objects. This list is produced by a source object. This
    allows us to not implement the recombination logic itself in each source
    type that needs it.
    """

    recombinations: list[api.RecombineFieldSpec]

    def __init__(self, source: api.Source, recombinations: list[api.RecombineFieldSpec]):
        super().__init__(source)
        self.recombinations = recombinations

    @property
    def properties(self) -> api.SourceProperties:
        # Don't pass the recombination specifications on: we're handling what
        # there is.
        return self.source.properties.update(
            recombine_fields=[],
        )

    def use_geometry(self, geometry: RecombinedField[F]) -> None:
        self.source.use_geometry(geometry.wrapped_field)

    def basis_of(self, field: RecombinedField[F]) -> B:
        return self.source.basis_of(field.wrapped_field)

    def geometries(self, basis: B) -> Iterator[RecombinedField]:
        for field in self.source.geometries(basis):
            yield RecombinedField(name=field.name, sources=[field])

    def fields(self, basis: B) -> Iterator[RecombinedField]:
        # Collect all fields in the source object.
        in_fields = {field.name: field for field in self.source.fields(basis)}

        # Yield all fields to pass through faithfully
        for field in in_fields.values():
            yield RecombinedField(name=field.name, sources=[field])

        # Then yield all the recombined fields for this basis
        for spec in self.recombinations:
            if all(src in in_fields for src in spec.source_names):
                yield RecombinedField(
                    name=spec.new_name, sources=[in_fields[src] for src in spec.source_names]
                )

    def field_data(self, timestep: S, field: RecombinedField[F], zone: Z) -> FieldData[floating]:
        return FieldData.join_comps(self.source.field_data(timestep, src, zone) for src in field.sources)

    def field_updates(self, timestep: S, field: RecombinedField[F]) -> bool:
        return any(self.source.field_updates(timestep, src) for src in field.sources)
