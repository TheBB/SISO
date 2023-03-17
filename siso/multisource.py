from typing import Iterator, List, Optional, Sequence

from attrs import define
from numpy import floating
from typing_extensions import Self

from .api import Basis, Field, ReaderSettings, Source, SourceProperties, Step, Zone
from .topology import Topology
from .util import FieldData, bisect


@define
class MultiSourceStep:
    index: int
    original: Step
    source: Source

    @property
    def value(self) -> Optional[float]:
        return self.original.value


class MultiSource(Source):
    sources: Sequence[Source]
    maxindex: List[int]

    def __init__(self, sources: Sequence[Source]):
        self.sources = sources
        self.maxindex = []

    def __enter__(self) -> Self:
        for src in self.sources:
            src.__enter__()
        return self

    def __exit__(self, *args) -> None:
        for src in self.sources:
            src.__exit__(*args)

    @property
    def properties(self) -> SourceProperties:
        return self.sources[0].properties.update(
            instantaneous=False,
        )

    def configure(self, settings: ReaderSettings) -> None:
        for source in self.sources:
            source.configure(settings)

    def source_at(self, index: int) -> Source:
        i = bisect.bisect_left(self.maxindex, index)
        return self.sources[i]

    def use_geometry(self, geometry: Field) -> None:
        for source in self.sources:
            source.use_geometry(geometry)

    def bases(self) -> Iterator[Basis]:
        return self.sources[0].bases()

    def basis_of(self, field: Field):
        return self.sources[0].basis_of(field)

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
        return source.topology(step.original, basis, zone)

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
