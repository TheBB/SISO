from __future__ import annotations

from functools import reduce
from typing import TYPE_CHECKING

from attrs import define

from siso.api import (
    Basis,
    Field,
    Step,
    Topology,
    Zone,
    impl_basis_of,
    impl_field_data,
    impl_field_updates,
    impl_fields,
    impl_geometries,
    impl_use_geometry,
)
from siso.util import FieldData

from .passthrough import PassthroughBSTZ, WrappedField

if TYPE_CHECKING:
    from collections.abc import Iterator

    from siso import api
    from siso.util.field_data import FloatFieldData


@define
class RecombinedField[F: Field](WrappedField[F]):
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


class Recombine[B: Basis, F: Field, S: Step, T: Topology, Z: Zone](
    PassthroughBSTZ[B, S, T, Z, F, RecombinedField[F]]
):
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

    @impl_use_geometry
    def use_geometry(self, geometry: RecombinedField[F]) -> None:
        self.source.use_geometry(geometry.wrapped_field)

    @impl_basis_of
    def basis_of(self, field: RecombinedField[F]) -> B:
        return self.source.basis_of(field.wrapped_field)

    @impl_geometries
    def geometries(self, basis: B) -> Iterator[RecombinedField]:
        for field in self.source.geometries(basis):
            yield RecombinedField(name=field.name, sources=[field])

    @impl_fields
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

    @impl_field_data
    def field_data(self, timestep: S, field: RecombinedField[F], zone: Z) -> FloatFieldData:
        return FieldData.join_comps(self.source.field_data(timestep, src, zone) for src in field.sources)

    @impl_field_updates
    def field_updates(self, timestep: S, field: RecombinedField[F]) -> bool:
        return any(self.source.field_updates(timestep, src) for src in field.sources)
