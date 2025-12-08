from __future__ import annotations

from typing import TYPE_CHECKING, Self, cast

from attrs import define

from .api import (
    Basis,
    Field,
    ReaderSettings,
    Source,
    SourceProperties,
    Step,
    Zone,
    impl_basis_of,
    impl_field_data,
    impl_field_updates,
    impl_fields,
    impl_geometries,
    impl_topology,
    impl_topology_updates,
)
from .util import bisect

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import TracebackType

    from siso.types import Float
    from siso.util.field_data import FloatFieldData

    from .topology import Topology


@define
class MultiSourceStep:
    index: int
    original: Step
    source: Source

    @property
    def value(self) -> Float | None:
        return self.original.value


class MultiSource(Source):
    sources: Sequence[Source]
    maxindex: list[int]

    def __init__(self, sources: Sequence[Source]):
        self.sources = sources
        self.maxindex = []

    def __enter__(self) -> Self:
        for src in self.sources:
            src.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        for src in self.sources:
            src.__exit__(exc_type, exc_val, exc_tb)

    @property
    def properties(self) -> SourceProperties:
        return self.sources[0].properties.update(
            instantaneous=False,
        )

    def configure(self, settings: ReaderSettings) -> None:
        for source in self.sources:
            source.configure(settings)

    def source_at(self, index: int) -> Source:
        i: int = bisect.bisect_left(self.maxindex, index)
        return self.sources[i]

    def use_geometry(self, geometry: Field) -> None:
        for source in self.sources:
            source.use_geometry(geometry)

    def bases(self) -> Iterator[Basis]:
        return self.sources[0].bases()

    @impl_basis_of
    def basis_of(self, field: Field) -> Basis:
        return cast("Basis", self.sources[0].basis_of(field))

    @impl_geometries
    def geometries(self, basis: Basis) -> Iterator[Field]:
        return self.sources[0].geometries(basis)

    @impl_fields
    def fields(self, basis: Basis) -> Iterator[Field]:
        return self.sources[0].fields(basis)

    def steps(self) -> Iterator[Step]:
        index = 0
        for i, src in enumerate(self.sources):
            for timestep in src.steps():
                yield MultiSourceStep(index=index, original=timestep, source=src)
                index += 1
            if len(self.maxindex) <= i:
                self.maxindex.append(index)

    def zones(self) -> Iterator[Zone]:
        yield from self.sources[0].zones()

    @impl_topology
    def topology(self, step: MultiSourceStep, basis: Basis, zone: Zone) -> Topology:
        source = self.source_at(step.index)
        return cast("Topology", source.topology(step.original, basis, zone))

    @impl_topology_updates
    def topology_updates(self, step: MultiSourceStep, basis: Basis) -> bool:
        source = self.source_at(step.index)
        return source.topology_updates(step.original, basis)

    @impl_field_data
    def field_data(self, step: MultiSourceStep, field: Field, zone: Zone) -> FloatFieldData:
        source = self.source_at(step.index)
        return source.field_data(step.original, field, zone)

    @impl_field_updates
    def field_updates(self, step: MultiSourceStep, field: Field) -> bool:
        return step.source.field_updates(step.original, field)

    def children(self) -> Iterator[Source]:
        yield from self.sources
