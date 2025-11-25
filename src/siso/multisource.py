from __future__ import annotations

from typing import TYPE_CHECKING, Self, cast

from attrs import define

from .api import Basis, Field, ReaderSettings, Source, SourceProperties, Step, Zone
from .util import FieldData, bisect

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import TracebackType

    from numpy import floating

    from .topology import Topology


@define
class MultiSourceStep:
    index: int
    original: Step
    source: Source

    @property
    def value(self) -> float | None:
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

    def basis_of(self, field: Field) -> Basis:
        return cast("Basis", self.sources[0].basis_of(field))

    def geometries(self, basis: Basis) -> Iterator[Field]:
        return self.sources[0].geometries(basis)

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

    def topology(self, step: MultiSourceStep, basis: Basis, zone: Zone) -> Topology:
        source = self.source_at(step.index)
        return cast("Topology", source.topology(step.original, basis, zone))

    def topology_updates(self, step: MultiSourceStep, basis: Basis) -> bool:
        source = self.source_at(step.index)
        return source.topology_updates(step.original, basis)

    def field_data(self, step: MultiSourceStep, field: Field, zone: Zone) -> FieldData[floating]:
        source = self.source_at(step.index)
        return source.field_data(step.original, field, zone)

    def field_updates(self, step: MultiSourceStep, field: Field) -> bool:
        return step.source.field_updates(step.original, field)

    def children(self) -> Iterator[Source]:
        yield from self.sources
