"""Module for Simra format writer."""

import numpy as np
from numpy.linalg import norm
import treelog as log
from scipy.io import FortranFile

from .. import config
from ..fields import Field
from ..geometry import Patch, StructuredTopology
from ..util import structured_cells
from .writer import Writer

from ..typing import Array, Array2D


def fix_orientation(data: Array, tol=1e-2) -> Array:
    a = data[1,0,0] - data[0,0,0]
    b = data[0,1,0] - data[0,0,0]
    c = data[0,0,1] - data[0,0,0]

    x = np.dot(c / norm(c), np.cross(a / norm(a), b / norm(b)))
    if x > tol:
        log.warning("Swapping horizontal axes for left-handed mesh")
        data = data.transpose((1, 0, 2, 3))
    return data


class SIMRAWriter(Writer):
    """Simra format writer."""

    writer_name = "SIMRA"

    @classmethod
    def applicable(cls, fmt: str) -> bool:
        return fmt == 'dat'

    def validate(self):
        config.require(multiple_timesteps=False)

    def update_geometry(self, geometry: Field, patch: Patch, data: Array2D):
        if not isinstance(patch.topology, StructuredTopology):
            raise TypeError("SIMRA writer does not support unstructured grids")
        nodeshape = tuple(s+1 for s in patch.topology.shape)
        data = data.reshape(*nodeshape, 3)
        data = fix_orientation(data)
        cellshape = tuple(s-1 for s in data.shape[:-1])
        cells = structured_cells(cellshape, 3) + 1
        cells[:,1], cells[:,3] = cells[:,3].copy(), cells[:,1].copy()
        cells[:,5], cells[:,7] = cells[:,7].copy(), cells[:,5].copy()

        # Compute macro elements
        rshape = tuple(c - 1 for c in cellshape)
        mcells = structured_cells(tuple(c - 1 for c in cellshape), 3).reshape(*rshape, -1) + 1
        mcells = mcells[::2, ::2, ::2, ...].transpose((1, 0, 2, 3))
        mcells = mcells.reshape(-1, 8)
        mcells[:,1], mcells[:,3] = mcells[:,3].copy(), mcells[:,1].copy()
        mcells[:,5], mcells[:,7] = mcells[:,7].copy(), mcells[:,5].copy()

        # Write single precision
        data = data.astype('f4')
        cells = cells.astype('u4')
        mcells = mcells.astype('u4')

        with FortranFile(self.outpath, 'w', header_dtype='u4') as f:
            f.write_record(np.array([
                data.size // 3, cells.size // 8,
                data.shape[1], data.shape[0], data.shape[2],
                mcells.size // 8,
            ], dtype='u4'))
            f.write_record(data.flatten())
            f.write_record(cells.flatten())
            f.write_record(mcells.flatten())
        log.user(self.outpath)

    def update_field(self, field: Field, patch: Patch, data: Array2D):
        log.warning("SIMRA writer ignores fields")
