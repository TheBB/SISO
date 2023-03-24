from itertools import count, islice
from typing import Generic, Iterator, List, Optional, Tuple, TypeVar, overload

from attrs import define
from numpy import floating

from .. import api, util
from .passthrough import PassthroughBFZ


@overload
def islice_flag(stop: Optional[int], /) -> Iterator[bool]:
    ...


@overload
def islice_flag(start: Optional[int], stop: Optional[int], step: Optional[int], /) -> Iterator[bool]:
    ...


def islice_flag(*args):
    counter = islice(count(), *args)
    try:
        next_index = next(counter)
    except StopIteration:
        return

    for i in count():
        yield i == next_index
        if i == next_index:
            try:
                next_index = next(counter)
            except StopIteration:
                return


T = TypeVar("T")


@overload
def islice_group(it: Iterator[T], stop: Optional[int], /) -> Iterator[List[T]]:
    ...


@overload
def islice_group(
    it: Iterator[T], start: Optional[int], stop: Optional[int], step: Optional[int], /
) -> Iterator[List[T]]:
    ...


def islice_group(it, *args):
    accum = []
    for item, flag in zip(it, islice_flag(*args)):
        accum.append(item)
        if flag:
            yield accum
            accum = []


B = TypeVar("B", bound=api.Basis)
F = TypeVar("F", bound=api.Field)
S = TypeVar("S", bound=api.Step)
Z = TypeVar("Z", bound=api.Zone)


@define
class GroupedStep(Generic[S]):
    index: int
    steps: List[S]

    @property
    def value(self) -> Optional[float]:
        return self.steps[-1].value


class GroupedTimeSource(PassthroughBFZ[B, F, Z, S, GroupedStep[S]], Generic[B, F, S, Z]):
    def topology(self, step: GroupedStep[S], basis: B, zone: Z) -> api.Topology:
        return self.source.topology(step.steps[-1], basis, zone)

    def topology_updates(self, step: GroupedStep[S], basis: B) -> bool:
        return any(self.source.topology_updates(s, basis) for s in step.steps)

    def field_data(self, step: GroupedStep[S], field: F, zone: Z) -> util.FieldData[floating]:
        return self.source.field_data(step.steps[-1], field, zone)

    def field_updates(self, step: GroupedStep[S], field: F) -> bool:
        return any(self.source.field_updates(s, field) for s in step.steps)


class StepSlice(GroupedTimeSource[B, F, S, Z]):
    arguments: Tuple[Optional[int]]
    explicit_instantaneous: bool

    def __init__(
        self,
        source: api.Source[B, F, S, Z],
        arguments: Tuple[Optional[int]],
        explicit_instantaneous: bool = False,
    ):
        super().__init__(source)
        self.arguments = arguments
        self.explicit_instantaneous = explicit_instantaneous

    @property
    def properties(self) -> api.SourceProperties:
        props = self.source.properties
        if self.explicit_instantaneous:
            return props.update(instantaneous=True)
        return props

    def steps(self) -> Iterator[GroupedStep[S]]:
        for i, times in enumerate(islice_group(self.source.steps(), *self.arguments)):
            yield GroupedStep(i, times)


class LastTime(GroupedTimeSource[B, F, S, Z]):
    @property
    def properties(self) -> api.SourceProperties:
        return self.source.properties.update(instantaneous=True)

    def steps(self) -> Iterator[GroupedStep[S]]:
        steps = list(self.source.steps())
        yield GroupedStep(0, steps)
