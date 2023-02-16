from dataclasses import dataclass

from .api import Source, TimeStep, Field, SourceProperties
from .field import FieldData
from .topology import Topology
from .util import bisect
from .zone import Zone

from typing import (
    Iterator,
    List,
    Optional,
    Sequence,
)


@dataclass
class MultiSourceTimeStep:
    index: int
    original: TimeStep

    @property
    def time(self) -> Optional[float]:
        return self.original.time


class MultiSource(Source):
    sources: Sequence[Source]
    maxindex: List[int]

    def __init__(self, sources: Sequence[Source]):
        self.sources = sources
        self.maxindex = []

    def __enter__(self):
        for src in self.sources:
            src.__enter__()

    def __exit__(self, *args):
        for src in self.sources:
            src.__exit__(*args)

    @property
    def properties(self) -> SourceProperties:
        return self.sources[0].properties.update(
            instantaneous=False,
        )

    def source_at(self, index: int) -> Source:
        i = bisect.bisect_left(self.maxindex, index)
        return self.sources[i]

    def fields(self) -> Iterator[Field]:
        yield from self.sources[0].fields()

    def timesteps(self) -> Iterator[TimeStep]:
        index = 0
        for i, src in enumerate(self.sources):
            for timestep in src.timesteps():
                yield MultiSourceTimeStep(index=index, original=timestep)
                index += 1
            if len(self.maxindex) <= i:
                self.maxindex.append(index)

    def zones(self) -> Iterator[Zone]:
        yield from self.sources[0].zones()

    def topology(self, timestep: MultiSourceTimeStep, field: Field, zone: Zone) -> Topology:
        source = self.source_at(timestep.index)
        return source.topology(timestep.original, field, zone)

    def field_data(self, timestep: MultiSourceTimeStep, field: Field, zone: Zone) -> FieldData:
        source = self.source_at(timestep.index)
        return source.field_data(timestep.original, field, zone)
