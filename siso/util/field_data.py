from __future__ import annotations

import logging
import sys
from typing import Generic, Iterable, List, Optional, Tuple, TypeVar, Union, overload

import numpy as np
from attrs import define
from numpy import floating, integer, number
from numpy.typing import DTypeLike, NDArray
from scipy.spatial.transform import Rotation

from ..zone import Coords


try:
    from vtkmodules.util.numpy_support import numpy_to_vtk
    from vtkmodules.vtkCommonCore import vtkDataArray

    HAS_VTK = True
except ImportError:
    HAS_VTK = False


T = TypeVar("T", bound=number)
S = TypeVar("S", bound=number)
Index = Union[int, slice, None, NDArray[integer]]
Indices = Union[Index, Tuple[Index, ...]]


def ensure_2d_right(array: NDArray[T]) -> NDArray[T]:
    if array.ndim < 2:
        return array.reshape(-1, 1)
    return array


def ensure_2d_left(array: NDArray[T]) -> NDArray[T]:
    if array.ndim < 2:
        return array.reshape(1, -1)
    return array


@define
class FieldData(Generic[T]):
    data: NDArray[T]

    def __attrs_post_init__(self) -> None:
        assert self.data.ndim == 2

    @overload
    @staticmethod
    def concat(other: Iterable[Union[FieldData[T], NDArray[T]]], /) -> FieldData[T]:
        ...

    @overload
    @staticmethod
    def concat(*other: Union[FieldData[T], NDArray[T]]) -> FieldData[T]:
        ...

    @staticmethod
    def concat(*other):
        iterable = other if isinstance(other[0], (FieldData, np.ndarray)) else other[0]
        data = np.hstack([ensure_2d_right(x.numpy() if isinstance(x, FieldData) else x) for x in iterable])
        return FieldData(data)

    @overload
    @staticmethod
    def join(other: Iterable[Union[FieldData[T], NDArray[T]]], /) -> FieldData[T]:
        ...

    @overload
    @staticmethod
    def join(*other: Union[FieldData[T], NDArray[T]]) -> FieldData[T]:
        ...

    @staticmethod
    def join(*other):
        iterable = other if isinstance(other[0], (FieldData, np.ndarray)) else other[0]
        data = np.vstack([ensure_2d_left(x.numpy() if isinstance(x, FieldData) else x) for x in iterable])
        return FieldData(data)

    @overload
    @staticmethod
    def from_iter(
        iterable: Iterable[Iterable[Union[float, int]]], dtype: DTypeLike = float
    ) -> FieldData[floating]:
        ...

    @overload
    @staticmethod
    def from_iter(iterable: Iterable[Iterable[T]], dtype: DTypeLike = float) -> FieldData[T]:
        ...

    @staticmethod
    def from_iter(iterable, dtype: DTypeLike = float):
        ntuples = 0

        def values():
            nonlocal ntuples
            for value in iter(iterable):
                ntuples += 1
                yield from value

        array = np.fromiter(values(), dtype=dtype)
        return FieldData(array.reshape(ntuples, -1))

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def ncomps(self) -> int:
        return self.data.shape[-1]

    @property
    def ndofs(self) -> int:
        return self.data.shape[0]

    @property
    def components(self) -> Iterable[NDArray[T]]:
        return self.data.T

    @property
    def vectors(self) -> Iterable[NDArray[T]]:
        return self.data

    def __getitem__(self, indices: Indices) -> FieldData[T]:
        return FieldData(self.data[indices])

    def mean(self) -> NDArray[T]:
        return self.data.mean(axis=0)

    def slice(self, index: Union[int, List[int]]) -> FieldData[T]:
        if isinstance(index, int):
            return FieldData(data=self.data[:, index : index + 1])
        return FieldData(data=self.data[:, index])

    def nan_filter(self, fill: Optional[T] = None) -> FieldData[T]:
        i = np.where(np.isnan(self.data))
        if len(i[0]) > 0:
            logging.warning(f"NaN values set to {fill}")
            self.data[i] = fill if fill is not None else np.zeros((), dtype=self.data.dtype)
        return self

    def ensure_ncomps(self, ncomps: int, allow_scalar: bool = False, pad_right: bool = True) -> FieldData[T]:
        if self.data.shape[-1] == 1 and allow_scalar:
            return self
        if self.data.shape[-1] >= ncomps:
            return self
        new_comps = ncomps - self.data.shape[-1]
        filler = np.zeros((self.data.shape[0], new_comps), dtype=self.data.dtype)
        to_stack = (self.data, filler) if pad_right else (filler, self.data)
        return FieldData(data=np.hstack(to_stack))

    def ensure_native(self) -> FieldData[T]:
        if self.data.dtype.byteorder in ("=", sys.byteorder):
            return self
        return FieldData(self.data.byteswap().newbyteorder())

    def corners(self: FieldData[floating], shape: Tuple[int, ...]) -> Coords:
        temp = self.data.reshape(*shape, -1)
        corners = temp[tuple(slice(None, None, j - 1) for j in temp.shape[:-1])]
        corners = corners.reshape(-1, self.ncomps)
        return tuple(tuple(corner) for corner in corners)

    def collapse_weights(self: FieldData[floating]) -> FieldData[floating]:
        data = self.data[..., :-1] / self.data[..., -1:]
        return FieldData(data)

    def transpose(self, shape: Tuple[int, ...], transposition: Tuple[int, ...]) -> FieldData[T]:
        return FieldData(
            self.data.reshape(*shape, -1)
            .transpose(*transposition, len(transposition))
            .reshape(self.data.shape)
        )

    def swap_components(self, i: int, j: int) -> FieldData[T]:
        self.data[:,i], self.data[:,j] = self.data[:,j].copy(), self.data[:,i].copy()
        return self

    @overload
    def constant_like(
        self, value: int, ndofs: Optional[int] = None, ncomps: Optional[int] = None, dtype: DTypeLike = None
    ) -> FieldData[integer]:
        ...

    @overload
    def constant_like(
        self, value: float, ndofs: Optional[int] = None, ncomps: Optional[int] = None, dtype: DTypeLike = None
    ) -> FieldData[floating]:
        ...

    def constant_like(self, value, ndofs=None, ncomps=None, dtype=None):
        retval = np.ones_like(
            self.data,
            shape=(
                ndofs or self.ndofs,
                ncomps or self.ncomps,
            ),
            dtype=dtype,
        )
        retval.fill(value)
        return FieldData(retval)

    def trigonometric(self) -> FieldData[floating]:
        retval = np.zeros_like(self.data, shape=(self.ndofs, 4))
        lon, lat, *_ = self.components
        retval[:, 0] = np.cos(np.deg2rad(lon))
        retval[:, 1] = np.cos(np.deg2rad(lat))
        retval[:, 2] = np.sin(np.deg2rad(lon))
        retval[:, 3] = np.sin(np.deg2rad(lat))
        return FieldData(retval)

    def spherical_to_cartesian(self) -> FieldData[floating]:
        clon, clat, slon, slat = self.trigonometric().components
        retval = FieldData.concat(clon * clat, slon * clat, slat)
        if self.ncomps > 2:
            retval.data *= self.data[:, 2]
        return retval

    def cartesian_to_spherical(self, with_radius: bool = True) -> FieldData[floating]:
        x, y, z = self.components
        lon = np.rad2deg(np.arctan2(y, x))
        lat = np.rad2deg(np.arctan(z / np.sqrt(x**2 + y**2)))

        if not with_radius:
            return FieldData.concat(lon, lat)

        radius = np.sqrt(x**2 + y**2 + z**2)
        return FieldData.concat(lon, lat, radius)

    def spherical_to_cartesian_vector_field(self, coords: FieldData[floating]) -> FieldData[floating]:
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

    def cartesian_to_spherical_vector_field(self, coords: FieldData[floating]) -> FieldData[floating]:
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

    def rotate(self, rotation: Rotation) -> FieldData[floating]:
        return FieldData(rotation.apply(self.data))

    def numpy(self, *shape: int) -> NDArray[T]:
        if not shape:
            return self.data
        return self.data.reshape(*shape, self.ncomps)

    def vtk(self) -> vtkDataArray:
        assert HAS_VTK
        return numpy_to_vtk(self.data, deep=1)

    def __add__(self, other) -> FieldData:
        if isinstance(other, FieldData):
            return FieldData(self.data + other.data)
        return FieldData(self.data + other)

    @overload
    def __mul__(
        self: FieldData[integer], other: Union[int, NDArray[integer], FieldData[integer]]
    ) -> FieldData[integer]:
        ...

    @overload
    def __mul__(
        self: FieldData[floating], other: Union[int, float, NDArray[number], FieldData[number]]
    ) -> FieldData[floating]:
        ...

    def __mul__(self, other):
        if isinstance(other, FieldData):
            return FieldData(self.data * other.data)
        return FieldData(self.data * other)

    def __floordiv__(
        self: FieldData[integer], other: Union[int, NDArray[integer], FieldData[integer]]
    ) -> FieldData[integer]:
        if isinstance(other, FieldData):
            return FieldData(self.data // other.data)
        return FieldData(self.data // other)

    def __truediv__(self, other) -> FieldData:
        if isinstance(other, FieldData):
            return FieldData(self.data / other.data)
        return FieldData(self.data / other)
