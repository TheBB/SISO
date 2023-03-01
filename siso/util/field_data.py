from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from numbers import Number
from typing import Iterable, List, Tuple, Union

import numpy as np
from numpy.typing import DTypeLike

from ..zone import Coords


try:
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkDataArray

    HAS_VTK = True
except ImportError:
    HAS_VTK = False


@dataclass
class FieldData:
    data: np.ndarray

    def __post_init__(self) -> None:
        assert self.data.ndim == 2

    @staticmethod
    def concat(other: Iterable[FieldData]) -> FieldData:
        data = np.hstack([data.numpy() for data in other])
        return FieldData(data)

    @staticmethod
    def from_iter(iterable: Iterable[Iterable], dtype: DTypeLike = float) -> FieldData:
        ntuples = 0

        def values():
            nonlocal ntuples
            for value in iter(iterable):
                ntuples += 1
                yield from value

        array = np.fromiter(values(), dtype=dtype)
        return FieldData(array.reshape(ntuples, -1))

    @property
    def ncomps(self) -> int:
        return self.data.shape[-1]

    @property
    def ndofs(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, index: int) -> np.ndarray:
        return self.data[index, :]

    def join(self, other: FieldData) -> FieldData:
        return FieldData(np.vstack((self.data, other.data)))

    def slice(self, index: Union[int, List[int]]) -> FieldData:
        if isinstance(index, int):
            return FieldData(data=self.data[:, index : index + 1])
        return FieldData(data=self.data[:, index])

    def nan_filter(self, fill: float = 0.0) -> FieldData:
        i = np.where(np.isnan(self.data))
        if len(i[0]) > 0:
            logging.warning(f"NaN values set to {fill}")
            self.data[i] = fill
        return self

    def ensure_ncomps(self, ncomps: int, allow_scalar: bool = False, pad_right: bool = True) -> FieldData:
        if self.data.shape[-1] == 1 and allow_scalar:
            return self
        if self.data.shape[-1] >= ncomps:
            return self
        new_comps = ncomps - self.data.shape[-1]
        filler = np.zeros((self.data.shape[0], new_comps), dtype=self.data.dtype)
        to_stack = (self.data, filler) if pad_right else (filler, self.data)
        return FieldData(data=np.hstack(to_stack))

    def ensure_native(self) -> FieldData:
        if self.data.dtype.byteorder in ("=", sys.byteorder):
            return self
        return FieldData(self.data.byteswap().newbyteorder())

    def corners(self, shape: Tuple[int, ...]) -> Coords:
        temp = self.data.reshape(*shape, -1)
        corners = temp[tuple(slice(None, None, j - 1) for j in temp.shape[:-1])]
        corners = corners.reshape(-1, self.ncomps)
        return tuple(tuple(corner) for corner in corners)

    def translate(self, delta: np.ndarray) -> FieldData:
        assert delta.ndim == 1
        return FieldData(self.data + delta)

    def collapse_weights(self) -> FieldData:
        data = self.data[..., :-1] / self.data[..., -1:]
        return FieldData(data)

    def transpose(self, shape: Tuple[int, ...], transposition: Tuple[int, ...]) -> FieldData:
        return FieldData(
            self.data.reshape(*shape, -1)
            .transpose(*transposition, len(transposition))
            .reshape(self.data.shape)
        )

    def numpy(self) -> np.ndarray:
        return self.data

    def vtk(self) -> vtkDataArray:
        assert HAS_VTK
        return numpy_to_vtk(self.data, deep=1)

    def __add__(self, other) -> FieldData:
        if isinstance(other, FieldData):
            return FieldData(self.data + other.data)
        return NotImplemented

    def __truediv__(self, other) -> FieldData:
        if isinstance(other, Number):
            return FieldData(self.data / other)
        if isinstance(other, FieldData):
            return FieldData(self.data / other.data)
        return NotImplemented
