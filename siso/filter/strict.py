from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from siso import api, util
from siso.api import B, DiscreteTopology, F, S, T, Z

from .passthrough import PassthroughAll

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy import floating

    from siso.util import FieldData


class Strict(PassthroughAll[B, F, S, T, Z]):
    """Filter that changes nothing, but checks some invariants. Use for
    debugging. Should be always used in tests.
    """

    # Keep track of all fields to ensure they don't change
    field_specs: dict[str, F]

    # Keep track of the properties to ensure they don't change
    original_properties: api.SourceProperties

    geometry: F

    def __init__(self, source: api.Source[B, F, S, T, Z]):
        super().__init__(source)
        self.field_specs = {}
        self.original_properties = deepcopy(source.properties)

    @property
    def properties(self) -> api.SourceProperties:
        # Assert that the source properties never change
        properties = self.source.properties
        assert properties == self.original_properties
        return properties

    def use_geometry(self, geometry: F) -> None:
        super().use_geometry(geometry)
        self.geometry = deepcopy(geometry)

    def bases(self) -> Iterator[B]:
        bases = list(self.source.bases())
        if self.original_properties.single_basis:
            assert len(bases) == 1
        yield from bases

    def geometries(self, basis: B) -> Iterator[F]:
        for field in self.source.geometries(basis):
            assert field.is_geometry
            if field.name not in self.field_specs:
                self.field_specs[field.name] = field
            else:
                # Assert that the field is identical if seen more than once
                spec = self.field_specs[field.name]
                assert spec.cellwise == field.cellwise
                assert spec.splittable == field.splittable
                assert spec.name == field.name
                assert spec.type == field.type
            yield field

    def fields(self, basis: B) -> Iterator[F]:
        for field in self.source.fields(basis):
            assert not field.is_geometry
            if field.name not in self.field_specs:
                self.field_specs[field.name] = field
            else:
                # Assert that the field is identical if seen more than once
                spec = self.field_specs[field.name]
                assert spec.cellwise == field.cellwise
                assert spec.splittable == field.splittable
                assert spec.name == field.name
                assert spec.type == field.type
            yield field

    def steps(self) -> Iterator[S]:
        steps = list(self.source.steps())
        if self.original_properties.instantaneous:
            # Instantaneous sources only ever generate one step
            assert len(steps) == 1

        # Check that steps are properly ordered
        for a, b in util.pairwise(steps):
            assert b.index > a.index
            if b.value is not None and a.value is not None:
                assert b.value > a.value

        yield from steps

    def zones(self) -> Iterator[Z]:
        zones = list(self.source.zones())
        if self.original_properties.single_zoned:
            assert len(zones) == 1
        if self.original_properties.globally_keyed:
            assert all(isinstance(zone.key, int) for zone in zones)
        yield from zones

    def topology(self, step: S, basis: B, zone: Z) -> T:
        topology = self.source.topology(step, basis, zone)
        if self.original_properties.discrete_topology:
            # Sources marked with discrete topology can only produce discrete
            # topologies with degree 1
            assert isinstance(topology, DiscreteTopology)
            assert topology.degree == 1
        return topology

    def field_data(self, timestep: S, field: F, zone: Z) -> FieldData[floating]:
        if field.is_geometry:
            assert field.name == self.geometry.name
        data = self.source.field_data(timestep, field, zone)
        spec = self.field_specs[field.name]
        assert spec.cellwise == field.cellwise
        assert spec.splittable == field.splittable
        assert spec.name == field.name
        assert spec.type == field.type
        assert data.num_comps == field.num_comps
        return data
