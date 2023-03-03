from __future__ import annotations

import logging
import sys
from numbers import Number
from typing import Iterable, List, Tuple, Union, overload

import numpy as np
from numpy.typing import NDArray
from attrs import define
from numpy.typing import DTypeLike
from scipy.spatial.transform import Rotation

from ..zone import Coords


try:
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkDataArray

    HAS_VTK = True
except ImportError:
    HAS_VTK = False


def ensure_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim < 2:
        return array.reshape(-1, 1)
    return array


@define
class FieldData:
    data: np.ndarray

    def __post_init__(self) -> None:
        assert self.data.ndim == 2

    @overload
    @staticmethod
    def concat(other: Iterable[Union[FieldData, np.ndarray]], /) -> FieldData:
        ...

    @overload
    @staticmethod
    def concat(*other: Union[FieldData, np.ndarray]) -> FieldData:
        ...

    @staticmethod
    def concat(*other) -> FieldData:
        if isinstance(other[0], (FieldData, np.ndarray)):
            data = np.hstack([ensure_2d(x.numpy() if isinstance(x, FieldData) else x) for x in other])
        else:
            data = np.hstack([ensure_2d(x.numpy() if isinstance(x, FieldData) else x) for x in other[0]])
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

    @property
    def components(self) -> Iterable[np.ndarray]:
        return self.data.T

    @property
    def vectors(self) -> Iterable[np.ndarray]:
        return self.data

    def __getitem__(self, indices: Tuple[Union[int, slice, None, np.ndarray], ...]) -> FieldData:
        return FieldData(self.data[indices])

    def mean(self) -> np.ndarray:
        return np.mean(self.data, axis=0)

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

    def trigonometric(self) -> FieldData:
        retval = np.zeros_like(self.data, shape=(self.ndofs, 4))
        lon, lat, *_ = self.components
        retval[:, 0] = np.cos(np.deg2rad(lon))
        retval[:, 1] = np.cos(np.deg2rad(lat))
        retval[:, 2] = np.sin(np.deg2rad(lon))
        retval[:, 3] = np.sin(np.deg2rad(lat))
        return FieldData(retval)

    def spherical_to_cartesian(self) -> FieldData:
        clon, clat, slon, slat = self.trigonometric().components
        retval = FieldData.concat(clon * clat, slon * clat, slat)
        if self.ncomps > 2:
            retval.data *= self.data[:,2]
        return retval

    def cartesian_to_spherical(self, with_radius: bool = True) -> FieldData:
        x, y, z = self.components
        lon = np.rad2deg(np.arctan2(y, x))
        lat = np.rad2deg(np.arctan(z / np.sqrt(x**2 + y**2)))

        if not with_radius:
            return FieldData.concat(lon, lat)

        radius = np.sqrt(x**2 + y**2 + z**2)
        return FieldData.concat(lon, lat, radius)

    def spherical_to_cartesian_vector_field(self, coords: FieldData) -> FieldData:
        clon, clat, slon, slat = coords.trigonometric().components
        u, v, w = self.components
        retval = np.zeros_like(self.data)
        retval[..., 0] -= slon * u
        retval[..., 1] -= slat * slon * v
        retval[..., 2] += slat * w
        retval[..., 0] -= slat * clon * v
        retval[..., 0] += clat * clon * w
        retval[..., 1] += clon * u
        retval[..., 1] += clat * slon * w
        retval[..., 2] += clat * v
        return FieldData(retval)

    def cartesian_to_spherical_vector_field(self, coords: FieldData) -> FieldData:
        clon, clat, slon, slat = coords.trigonometric().components
        u, v, w = self.components
        retval = np.zeros_like(self.data)
        retval[..., 0] -= slon * u
        retval[..., 1] -= slat * slon * v
        retval[..., 2] += slat * w
        retval[..., 1] -= slat * clon * u
        retval[..., 2] += clat * clon * u
        retval[..., 0] += clon * v
        retval[..., 2] += clat * slon * v
        retval[..., 1] += clat * w
        return FieldData(retval)

    def rotate(self, rotation: Rotation) -> FieldData:
        return FieldData(rotation.apply(self.data))

    def numpy(self, *shape: int) -> np.ndarray:
        if not shape:
            return self.data
        return self.data.reshape(shape)

    def vtk(self) -> vtkDataArray:
        assert HAS_VTK
        return numpy_to_vtk(self.data, deep=1)

    def __add__(self, other) -> FieldData:
        if isinstance(other, FieldData):
            return FieldData(self.data + other.data)
        return FieldData(self.data + other)

    def __truediv__(self, other) -> FieldData:
        if isinstance(other, FieldData):
            return FieldData(self.data / other.data)
        return FieldData(self.data / other)
