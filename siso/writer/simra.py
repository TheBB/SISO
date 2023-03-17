import logging
from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.linalg import norm
from scipy.io import FortranFile
from typing_extensions import Self

from .. import api, util
from ..topology import StructuredTopology
from .api import Writer, WriterProperties, WriterSettings


B = TypeVar("B", bound=api.Basis)
F = TypeVar("F", bound=api.Field)
T = TypeVar("T", bound=api.Step)
Z = TypeVar("Z", bound=api.Zone)


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

    def __exit__(self, *args) -> None:
        self.data.__exit__(*args)
        logging.info(self.filename)

    @property
    def properties(self) -> WriterProperties:
        return WriterProperties(
            require_single_zone=True,
            require_tesselated=True,
            require_instantaneous=True,
        )

    def configure(self, settings: WriterSettings):
        self.f4_type = settings.endianness.f4_type()
        self.u4_type = settings.endianness.u4_type()

    def consume(self, source: api.Source[B, F, T, Z], geometry: F):
        timestep = next(source.steps())
        zone = next(source.zones())

        topology = source.topology(timestep, source.basis_of(geometry), zone)
        assert isinstance(topology, StructuredTopology)
        data = source.field_data(timestep, geometry, zone)

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
        cells = topology.cells.swap_components(1, 3).swap_components(5, 7).numpy().astype(self.u4_type) + 1

        macro_shape = tuple(c - 1 for c in topology.cellshape)
        macro_cells = (
            util.structured_cells(macro_shape, topology.pardim)
            .swap_components(1, 3)
            .swap_components(5, 7)
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
