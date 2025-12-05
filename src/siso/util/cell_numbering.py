from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import integer

from siso.api import CellOrdering, CellType

if TYPE_CHECKING:
    from numpy.typing import NDArray

NUMBERINGS: dict[tuple[CellType, int], dict[CellOrdering, NDArray[integer]]] = {
    # Linear line
    (CellType.Line, 1): {
        CellOrdering.Ifem: np.arange(2),
        CellOrdering.Siso: np.arange(2),
        CellOrdering.Vtk: np.arange(2),
    },
    # Linear triangle
    (CellType.Triangle, 1): {
        CellOrdering.Ifem: np.arange(3),
        CellOrdering.Siso: np.arange(3),
        CellOrdering.Vtk: np.arange(3),
    },
    # Bilinear quadrilateral
    (CellType.Quadrilateral, 1): {
        CellOrdering.Ifem: np.arange(4).reshape(2, 2),
        CellOrdering.Siso: np.arange(4).reshape(2, 2),
        CellOrdering.Vtk: np.array([0, 1, 3, 2]).reshape(2, 2).transpose(),
    },
    # Bilinear hexahedron
    (CellType.Hexahedron, 1): {
        CellOrdering.Ifem: np.arange(8).reshape(2, 2, 2).transpose(),
        CellOrdering.Simra: np.array([0, 3, 1, 2, 4, 7, 5, 6]).reshape(2, 2, 2).transpose(),
        CellOrdering.Siso: np.arange(8).reshape(2, 2, 2),
        CellOrdering.Vtk: np.array([0, 1, 3, 2, 4, 5, 7, 6]).reshape(2, 2, 2).transpose(),
    },
    # Biquadratic hexahedron
    (CellType.Hexahedron, 2): {
        CellOrdering.Ifem: np.arange(27).reshape(3, 3, 3).transpose(),
        CellOrdering.Siso: np.arange(27).reshape(3, 3, 3),
        CellOrdering.Vtk: np.array(
            [
                0,
                8,
                1,
                11,
                24,
                9,
                3,
                10,
                2,
                16,
                22,
                17,
                20,
                26,
                21,
                19,
                23,
                18,
                4,
                12,
                5,
                15,
                25,
                13,
                7,
                14,
                6,
            ]
        )
        .reshape(3, 3, 3)
        .transpose(),
    },
}


def _invert(permutation: list[int]) -> list[int]:
    retval = [0] * len(permutation)
    for i, j in enumerate(permutation):
        retval[j] = i
    return retval


def _compose(a: list[int], b: list[int]) -> list[int]:
    return [a[j] for j in b]


def permute_from(celltype: CellType, degree: int, ordering: CellOrdering) -> list[int]:
    return list(NUMBERINGS[celltype, degree][ordering].flatten())


def permute_to(celltype: CellType, degree: int, ordering: CellOrdering) -> list[int]:
    return _invert(permute_from(celltype, degree, ordering))


def permute_from_to(celltype: CellType, degree: int, src: CellOrdering, tgt: CellOrdering) -> list[int]:
    return _compose(
        permute_from(celltype, degree, src),
        permute_to(celltype, degree, tgt),
    )
