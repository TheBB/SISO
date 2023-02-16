from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import logging

import numpy as np

try:
    from vtkmodules.vtkCommonCore import vtkDataArray
    from vtkmodules.util.numpy_support import numpy_to_vtk
    HAS_VTK = True
except ImportError:
    HAS_VTK = False


class FieldType(Enum):
    Geometry = auto()
    Displacement = auto()
    Generic = auto()


@dataclass
class Field:
    name: str
    type: FieldType = FieldType.Generic
    ncomps: int = 1
    cellwise: bool = False


@dataclass
class FieldData:
    data: np.ndarray

    def __post_init__(self):
        assert self.data.ndim == 2

    @property
    def ncomps(self) -> int:
        return self.data.shape[-1]

    @property
    def ndofs(self) -> int:
        return self.data.shape[0]

    def join(self, other: FieldData) -> FieldData:
        return FieldData(
            data=np.vstack((self.data, other.data))
        )

    def nan_filter(self, fill: float = 0.0) -> FieldData:
        i = np.where(np.isnan(self.data))
        if len(i[0]) > 0:
            logging.warning('NaN values set to zero')
            self.data[i] = 0.0
        return self

    def ensure_ncomps(self, ncomps: int, allow_scalar: bool = False) -> FieldData:
        if self.data.shape[-1] == 1 and allow_scalar:
            return self
        if self.data.shape[-1] >= ncomps:
            return self
        new_comps = ncomps - self.data.shape[-1]
        filler = np.zeros((self.data.shape[0], new_comps), dtype=self.data.dtype)
        return FieldData(data=np.hstack((self.data, filler)))

    def numpy(self) -> np.ndarray:
        return self.data

    def vtk(self) -> vtkDataArray:
        assert HAS_VTK
        return numpy_to_vtk(self.data, deep=1)
