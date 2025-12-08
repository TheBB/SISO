from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np
from more_itertools import peekable

from siso.api import CellType, ZoneShape
from siso.topology import UnstructuredTopology
from siso.types import f64d, i32d
from siso.util import FieldData

from .puregeometry import PureGeometry, PureGeometryZone

if TYPE_CHECKING:
    from collections.abc import Iterator

    from siso.types import Matrix


IGNORED_CARDS = {
    "HDF5OUT",
    "LOAD",
    "MAT1",
    "PARAM",
    "PLOAD4",
    "PSHELL",
    "RBE2",
    "SPC1",
    "SPCADD",
}


@dataclass(slots=True)
class Card:
    kind: str
    args: list[str]


class GrowableMatrix[D: np.dtype]:
    __slots__ = ["data", "highest_used"]

    data: Matrix[D]
    highest_used: int

    def __init__(self, data: Matrix[D]) -> None:
        self.data = data
        self.highest_used = -1

    def __setitem__(self, index: int, data: Any) -> None:
        while self.data.shape[0] <= index:
            self.data.resize((2 * self.data.shape[0], self.data.shape[1]))
        self.data[index, : len(data)] = data
        self.highest_used = max(self.highest_used, index)


class GrowableGridArray(GrowableMatrix[f64d]):
    def __init__(self, ncomps: int) -> None:
        super().__init__(np.empty((10, ncomps), dtype=float))

    def finalize(self) -> FieldData[f64d]:
        return FieldData(self.data[: self.highest_used + 1])


class GrowableCellArray(GrowableMatrix[i32d]):
    def __init__(self, ncomps: int) -> None:
        super().__init__(np.empty((10, ncomps + 1), dtype=int))

    def __setitem__(self, index: int, data: Any) -> None:
        super().__setitem__(index, [len(data), *data])

    def triangles(self) -> FieldData[i32d]:
        (indices,) = np.where(self.data[: self.highest_used + 1, 0] == 3)
        return FieldData(self.data[indices, 1:4])

    def quads(self) -> FieldData[i32d]:
        (indices,) = np.where(self.data[: self.highest_used + 1, 0] == 4)
        return FieldData(self.data[indices, 1:5])


def noncomment_lines(f: TextIO) -> Iterator[str]:
    for line in f:
        if not line.startswith("$"):
            yield line


def is_continuation(line: str) -> bool:
    return line[:8].strip().removesuffix("*") == ""


def decompose(line: str) -> list[str]:
    card_type = line[:8].strip()
    double_format = card_type.endswith("*")
    card_type = card_type.removesuffix("*")

    args: tuple[str, ...]
    if double_format:
        args = line[8:24], line[24:40], line[40:56], line[56:72]
    else:
        args = (
            line[8:16],
            line[16:24],
            line[24:32],
            line[32:40],
            line[40:48],
            line[48:56],
            line[56:64],
            line[64:72],
        )

    stripped_args = (arg.strip() for arg in args)
    return [card_type, *stripped_args]


def cards(f: TextIO) -> Iterator[Card]:
    lines = peekable(noncomment_lines(f))

    for line in lines:
        if line.startswith("BEGIN BULK"):
            break

    while True:
        try:
            line = next(lines)
        except StopIteration:
            return
        args = decompose(line)
        try:
            while is_continuation(lines.peek()):
                args.extend(decompose(next(lines))[1:])
        except StopIteration:
            continue

        yield Card(args[0], args[1:])


class Nastran(PureGeometry[UnstructuredTopology]):
    """Reader class for Nastran input files."""

    def __enter__(self) -> Nastran:
        cells = GrowableCellArray(4)
        grid = GrowableGridArray(3)

        with self.filename.open() as f:
            for card in cards(f):
                if card.kind in IGNORED_CARDS:
                    continue

                if card.kind == "CQUAD4":
                    cells[int(card.args[0]) - 1] = [int(c) - 1 for c in card.args[2:6]]
                elif card.kind == "CTRIA3":
                    cells[int(card.args[0]) - 1] = [int(c) - 1 for c in card.args[2:5]]
                elif card.kind == "GRID":
                    grid[int(card.args[0]) - 1] = [
                        float(re.sub("(.)([-+])", r"\1e\2", c)) for c in card.args[2:5]
                    ]
                else:
                    print(card)
                    assert False

        points = grid.finalize()

        # triangles = cells.triangles()
        # if triangles.num_dofs > 0:
        #     corners = points.slice_dofs(triangles.data.flatten()).bounding_corners()
        #     self.zone_data.append(
        #         PureGeometryZone(
        #             corners,
        #             points,
        #             UnstructuredTopology(points.num_dofs, triangles, CellType.Triangle, 1),
        #             ZoneShape.Shapeless,
        #         )
        #     )

        quads = cells.quads()

        def interchange(i: int, j: int) -> None:
            quads.data[:, i], quads.data[:, j] = quads.data[:, j].copy(), quads.data[:, i].copy()

        interchange(2, 3)
        if quads.num_dofs > 0:
            corners = points.slice_dofs(quads.data.flatten()).bounding_corners()
            self.zone_data.append(
                PureGeometryZone(
                    corners,
                    points,
                    UnstructuredTopology(points.num_dofs, quads, CellType.Quadrilateral, 1),
                    ZoneShape.Shapeless,
                )
            )

        return self
