from __future__ import annotations

from itertools import count, islice
from typing import TYPE_CHECKING, overload

from attrs import define

from siso.api import (
    Basis,
    Field,
    Step,
    Topology,
    Zone,
    impl_field_data,
    impl_field_updates,
    impl_topology,
    impl_topology_updates,
)

from .passthrough import PassthroughBFTZ

if TYPE_CHECKING:
    from collections.abc import Iterator

    from siso import api
    from siso.types import Float
    from siso.util.field_data import FloatFieldData


@overload
def islice_flag(stop: int | None, /) -> Iterator[bool]: ...


@overload
def islice_flag(start: int | None, stop: int | None, step: int | None, /) -> Iterator[bool]: ...


def islice_flag(*args):  # type: ignore[no-untyped-def]
    """Version of `itertools.islice` that, instead of slicing an iterator to
    produce elements, yields a boolean indicating whether each element should be
    picked or not. E.g., this:

    ```
    islice(iterator, *args)
    ```

    is equivalent to:

    ```
    (e for e, f in zip(iterator, islice_flag(*args)) if f)
    ```
    """

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


@overload
def islice_group[Q](it: Iterator[Q], stop: int | None, /) -> Iterator[list[Q]]: ...


@overload
def islice_group[Q](
    it: Iterator[Q], start: int | None, stop: int | None, step: int | None, /
) -> Iterator[list[Q]]: ...


def islice_group(it, *args):  # type: ignore[no-untyped-def]
    """Version of `itertools.islice` that, for all 'picked' elements, yields a
    list of all unpicked elements since the previously picked element (or the
    beginning). E.g., if these iterators are given:

    ```
    iterator  # => A, B, C, ...
    islice(iterator, *args)  # => C, F, I, L, ...
    ```

    then

    ```
    islice_group(iterator, *args)
    # => [A, B, C]
    # => [E, D, F]
    # => [G, H, I]
    # => [J, K, L]
    # ...
    ```
    """

    accum = []
    for item, flag in zip(it, islice_flag(*args)):
        accum.append(item)
        if flag:
            yield accum
            accum = []


@define
class GroupedStep[S: Step]:
    """A step composed of a multiple steps in sequence."""

    index: int
    steps: list[S]

    @property
    def value(self) -> Float | None:
        # Act as the last step of the group.
        return self.steps[-1].value


class GroupedTimeSource[B: Basis, F: Field, S: Step, T: Topology, Z: Zone](
    PassthroughBFTZ[B, F, T, Z, S, GroupedStep[S]]
):
    """Base class for all filters that group timesteps into `GroupedStep`."""

    @impl_topology
    def topology(self, step: GroupedStep[S], basis: B, zone: Z) -> T:
        return self.source.topology(step.steps[-1], basis, zone)

    @impl_topology_updates
    def topology_updates(self, step: GroupedStep[S], basis: B) -> bool:
        return any(self.source.topology_updates(s, basis) for s in step.steps)

    @impl_field_data
    def field_data(self, step: GroupedStep[S], field: F, zone: Z) -> FloatFieldData:
        return self.source.field_data(step.steps[-1], field, zone)

    @impl_field_updates
    def field_updates(self, step: GroupedStep[S], field: F) -> bool:
        return any(self.source.field_updates(s, field) for s in step.steps)


class StepSlice[B: Basis, F: Field, S: Step, T: Topology, Z: Zone](GroupedTimeSource[B, F, S, T, Z]):
    """Filter that slices a sequence of timesteps, just like `itertools.islice`
    would.

    Parameters:
    - source: the data source.
    - arguments: tuple of arguments to pass to `islice`.
    - explicit_instantaneous: true if the source should be marked explicitly as
        instantaneous (only one timestep).
    """

    arguments: tuple[int | None]
    explicit_instantaneous: bool

    def __init__(
        self,
        source: api.Source[B, F, S, T, Z],
        arguments: tuple[int | None],
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


class LastTime[B: Basis, F: Field, S: Step, T: Topology, Z: Zone](GroupedTimeSource[B, F, S, T, Z]):
    """Filter that returns only the last timestep in a data source."""

    @property
    def properties(self) -> api.SourceProperties:
        return self.source.properties.update(instantaneous=True)

    def steps(self) -> Iterator[GroupedStep[S]]:
        steps = list(self.source.steps())
        yield GroupedStep(0, steps)
