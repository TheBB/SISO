"""This module implements filters for splitting fields by component."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic

from attrs import define

from siso import api
from siso.api import B, F, S, T, Z

from .passthrough import PassthroughBSTZ, WrappedField

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy import floating

    from siso.util import FieldData


@define
class DecomposedField(WrappedField[F]):
    """Class for a 'decomposed' field: a field sources its data from
    another field, but using a subset of components.

    Parameters:
    - wrapped_field: the field to source from.
    - components: a list of indices to pick, or None (reproduce the source field
        faithfully).
    - splittable: true if this field can be split.
    - name: name of the new field.
    """

    wrapped_field: F
    components: list[int] | None
    splittable: bool
    name: str

    @property
    def type(self) -> api.FieldType:
        if self.components is not None:
            if len(self.components) > 1:
                # Pick more than one component: source field must be a vector
                assert isinstance(self.wrapped_field.type, api.Vector)
                return self.wrapped_field.type.update(num_comps=len(self.components))

            # Pick one component: convert field type to scalar
            return self.wrapped_field.type.as_scalar()

        # Reproduce faithfully: just pass through the field type
        return self.wrapped_field.type


class DecomposeBase(PassthroughBSTZ[B, S, T, Z, F, DecomposedField[F]], Generic[B, F, S, T, Z]):
    """Base class for decomposition filters."""

    def use_geometry(self, geometry: DecomposedField[F]) -> None:
        return self.source.use_geometry(geometry.wrapped_field)

    def basis_of(self, field: DecomposedField[F]) -> B:
        return self.source.basis_of(field.wrapped_field)

    def geometries(self, basis: B) -> Iterator[DecomposedField[F]]:
        # Geometries should never be decomposed
        for field in self.source.geometries(basis):
            yield DecomposedField(name=field.name, wrapped_field=field, components=None, splittable=False)

    def field_data(self, timestep: S, field: DecomposedField, zone: Z) -> FieldData[floating]:
        data = self.source.field_data(timestep, field.wrapped_field, zone)
        if field.components is not None:
            data = data.slice_comps(field.components)
        return data

    def field_updates(self, timestep: S, field: DecomposedField[F]) -> bool:
        return self.source.field_updates(timestep, field.wrapped_field)


class Decompose(DecomposeBase[B, F, S, T, Z]):
    """Decompose filter. This filter automatically decomposes all vector fields
    that are marked as splittable with up to three components.

    Decomposed fields are named abc_x, abc_y and abc_z, where 'abc' is the name
    of the original field.
    """

    def fields(self, basis: B) -> Iterator[DecomposedField[F]]:
        for field in self.source.fields(basis):
            # Always pass through the original field unchanged
            yield DecomposedField(name=field.name, wrapped_field=field, components=None, splittable=False)

            # Conditions for not decomposing a field
            if field.is_scalar or not field.splittable or field.num_comps > 3:
                continue

            for i, suffix in zip(range(field.num_comps), "xyz"):
                name = f"{field.name}_{suffix}"
                yield DecomposedField(name=name, wrapped_field=field, components=[i], splittable=False)


class Split(DecomposeBase[B, F, S, T, Z]):
    """Split filter. This filter decomposes fields as indicated by a list of
    `SplitFieldSpec` objects. This list is produced by a source object. This
    allows us to not implement the splitting logic itself in each source type
    that needs it.
    """

    splits: list[api.SplitFieldSpec]

    def __init__(self, source: api.Source, splits: list[api.SplitFieldSpec]):
        super().__init__(source)
        self.splits = splits

    @property
    def properties(self) -> api.SourceProperties:
        # Don't pass the splitting specifications on: we're handling what there is.
        return self.source.properties.update(
            split_fields=[],
        )

    def fields(self, basis: B) -> Iterator[DecomposedField[F]]:
        # Fields that should be destroyed (not passed on faithfully)
        to_destroy = {split.source_name for split in self.splits if split.destroy}

        # Collect all fields in the source object
        fields = {field.name: field for field in self.source.fields(basis)}

        # Yield all fields that should be passed through faithfully
        for field in fields.values():
            if field.name not in to_destroy:
                yield DecomposedField(
                    name=field.name, wrapped_field=field, components=None, splittable=field.splittable
                )

        # Yield all split fields
        for split in self.splits:
            yield DecomposedField(
                name=split.new_name,
                wrapped_field=fields[split.source_name],
                components=split.components,
                splittable=split.splittable,
            )
