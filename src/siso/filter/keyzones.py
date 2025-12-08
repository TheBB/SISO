from __future__ import annotations

import logging
from collections.abc import Iterator, MutableMapping
from functools import reduce
from operator import itemgetter
from typing import TYPE_CHECKING, cast

from siso import api
from siso.api import (
    Basis,
    Field,
    Point,
    Source,
    SourceProperties,
    Step,
    Topology,
    Zone,
    ZoneShape,
    impl_field_data,
    impl_topology,
)
from siso.util import bisect

from .passthrough import PassthroughBFST

if TYPE_CHECKING:
    from siso.types import Float
    from siso.util.field_data import FloatFieldData


class KeyZones[B: Basis, F: Field, S: Step, T: Topology, Z: Zone](PassthroughBFST[B, F, S, T, Z, Zone[int]]):
    manager: ZoneManager
    mapping: dict[Zone[int], Z]

    def __init__(self, source: Source):
        super().__init__(source)
        self.manager = ZoneManager()
        self.mapping = {}

    def validate_source(self) -> None:
        if self.source.properties.globally_keyed:
            raise api.Unexpected("KeyZones filter applied to globally keyed source")

    @property
    def properties(self) -> SourceProperties:
        return super().properties.update(
            globally_keyed=True,
        )

    def zones(self) -> Iterator[Zone[int]]:
        for zone in self.source.zones():
            new_zone = self.manager.lookup(zone)
            self.mapping[new_zone] = zone
            yield new_zone

    @impl_topology
    def topology(self, timestep: S, basis: B, zone: Zone[int]) -> T:
        return self.source.topology(timestep, basis, self.mapping[zone])

    @impl_field_data
    def field_data(self, timestep: S, field: F, zone: Zone[int]) -> FloatFieldData:
        return self.source.field_data(timestep, field, self.mapping[zone])


class ZoneManager:
    lut: VertexDict[set[int]]
    shapes: dict[int, ZoneShape]

    def __init__(self) -> None:
        self.lut = VertexDict()
        self.shapes = {}

    def lookup(self, zone: Zone) -> Zone[int]:
        keys = reduce(lambda x, y: x & y, (self.lut.get(pt, set()) for pt in zone.coords))
        if len(keys) >= 2:
            raise api.Unexpected("Multiple zone candidates found")

        if keys:
            key = next(iter(keys))
            if self.shapes[key] != zone.shape:
                raise api.Unexpected("Differing zone shapes for same global key")
        else:
            key = len(self.shapes)
            if key in self.shapes:
                raise api.Unexpected("New global zone key seen before")
            self.shapes[key] = zone.shape
            for pt in zone.coords:
                self.lut.setdefault(pt, set()).add(key)
            logging.debug(f"Local zone '{zone.key}' associated with new global zone {key}")

        return Zone(
            shape=zone.shape,
            coords=zone.coords,
            key=key,
        )


class VertexDict[Q](MutableMapping[Point, Q]):
    rtol: float
    atol: float

    _keys: list[Point | None]
    _values: list[Q | None]

    lut: dict[int, list[tuple[int, float]]]

    def __init__(self, rtol: float = 1e-5, atol: float = 1e-8):
        self.rtol = rtol
        self.atol = atol
        self._keys = []
        self._values = []
        self.lut = {}

    def _bounds(self, key: Float) -> tuple[Float, Float]:
        if key >= self.atol:
            return (
                (key - self.atol) / (1 + self.rtol),
                (key + self.atol) / (1 - self.rtol),
            )

        if key <= -self.atol:
            return (
                (key - self.atol) / (1 - self.rtol),
                (key + self.atol) / (1 + self.rtol),
            )

        return (
            (key - self.atol) / (1 - self.rtol),
            (key + self.atol) / (1 - self.rtol),
        )

    def _candidate(self, key: Point) -> int:
        candidates = None
        for coord, k in enumerate(key):
            lut = self.lut.setdefault(coord, [])
            minval, maxval = self._bounds(k)
            lo = bisect.bisect_left(lut, minval, key=itemgetter(1))
            hi = bisect.bisect_left(lut, maxval, key=itemgetter(1))
            if candidates is None:
                candidates = {i for i, _ in lut[lo:hi]}
            else:
                candidates &= {i for i, _ in lut[lo:hi]}
        if candidates is None:
            raise KeyError(key)
        for c in candidates:
            if self._keys[c] is not None:
                return c
        raise KeyError(key)

    def _insert(self, key: Point, value: Q) -> None:
        newindex = len(self._values)
        for coord, v in enumerate(key):
            lut = self.lut.setdefault(coord, [])
            bisect.insort(lut, (newindex, v), key=itemgetter(1))
        self._keys.append(key)
        self._values.append(value)

    def __setitem__(self, key: Point, value: Q) -> None:
        try:
            c = self._candidate(key)
            self._values[c] = value
        except KeyError:
            self._insert(key, value)

    def __getitem__(self, key: Point) -> Q:
        c = self._candidate(key)
        return cast("Q", self._values[c])

    def __delitem__(self, key: Point) -> None:
        try:
            i = self._candidate(key)
        except KeyError:
            return
        self._keys[i] = None
        self._values[i] = None

    def __iter__(self) -> Iterator[Point]:
        for key in self._keys:
            if key is not None:
                yield key

    def __len__(self) -> int:
        return len(self._values)
