from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Self

import numpy as np
from numpy.linalg import norm
from scipy.io import FortranFile

from siso import api, util
from siso.api import Basis, CellShape, Field, Step, Topology, Zone
from siso.topology import StructuredTopology
from siso.util import cell_numbering

from .api import Writer, WriterProperties, WriterSettings

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


class SimraWriter(Writer):
    filename: Path
    data: FortranFile

    f4_type: np.dtype
    u4_type: np.dtype

    def __init__(self, filename: Path):
        self.filename = filename

    def __enter__(self) -> Self:
        self.data = FortranFile(self.filename, "w", header_dtype=self.u4_type)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.data.__exit__(exc_type, exc_val, exc_tb)
        logging.info(self.filename)

    @property
    def properties(self) -> WriterProperties:
        return WriterProperties(
            require_single_zone=True,
            require_discrete_topology=True,
            require_instantaneous=True,
        )

    def configure(self, settings: WriterSettings) -> None:
        self.f4_type = settings.endianness.f4_type()
        self.u4_type = settings.endianness.u4_type()

    def consume[B: Basis, F: Field, S: Step, T: Topology, Z: Zone](
        self, source: api.Source[B, F, S, T, Z], geometry: F
    ) -> None:
        casted = source.cast_discrete_topology()
        step = casted.single_step()
        zone = casted.single_zone()

        topology = casted.topology(step, casted.basis_of(geometry), zone)
        assert isinstance(topology, StructuredTopology)
        data = casted.field_data(step, geometry, zone)

        nodes = data.numpy(*topology.nodeshape)
        a = nodes[1, 0, 0] - nodes[0, 0, 0]
        b = nodes[0, 1, 0] - nodes[0, 0, 0]
        c = nodes[0, 0, 1] - nodes[0, 0, 0]

        x = np.dot(c / norm(c), np.cross(a / norm(a), b / norm(b)))
        tol = 1e-2
        if x > tol:
            logging.warning(f"Swapping horizontal axes for left-handed mesh ({x:.2e} > {tol:.2e})")
            nodes = nodes.transpose((1, 0, 2, 3))
            topology = topology.transpose((1, 0, 2))

        nodes = nodes.astype(self.f4_type)
        cells = topology.cells_as(api.CellOrdering.Simra).numpy().astype(self.u4_type) + 1

        macro_shape = CellShape(tuple(c - 1 for c in topology.cellshape))
        permutation = cell_numbering.permute_to(
            api.CellType.Hexahedron, degree=1, ordering=api.CellOrdering.Simra
        )
        macro_cells = (
            util.structured_cells(macro_shape, topology.pardim)
            .permute_components(permutation)
            .numpy(*macro_shape)[::2, ::2, ::2, :]
            .transpose((1, 0, 2, 3))
            .reshape(-1, 8)
            .astype(self.u4_type)
        ) + 1

        self.data.write_record(
            np.array(
                [
                    nodes.size // 3,
                    cells.shape[0],
                    nodes.shape[1],
                    nodes.shape[0],
                    nodes.shape[2],
                    macro_cells.shape[0],
                ],
                dtype=self.u4_type,
            )
        )

        self.data.write_record(nodes.flatten())
        self.data.write_record(cells.flatten())
        self.data.write_record(macro_cells.flatten())
