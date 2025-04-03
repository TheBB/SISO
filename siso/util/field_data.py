"""This module implements the FieldData class. This is a wrapper around a 2D
numpy array with an interface that is tailored to our use case. The first axis
is 'spatial', generally per topological node but also per cell (for cellwise
fields). The second axis is component. It has length one for scalar fields,
length 2 or 3 for 2D or 3D flow fields, etc.

Throughout this module we will name the axes the 'dof' axis and the 'comp' or
'component' axis.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Generic, TypeVar, cast, overload

import numpy as np
from attrs import define
from numpy import floating, integer, number
from numpy.typing import DTypeLike, NDArray
from vtkmodules.util.numpy_support import numpy_to_vtk

from siso.api import NodeShape, Point, Points

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from typing import Any

    from scipy.spatial.transform import Rotation
    from vtkmodules.vtkCommonCore import vtkDataArray

T = TypeVar("T", bound=number)
S = TypeVar("S", bound=number)
Index = int | slice | None | NDArray[integer]
Indices = Index | tuple[Index, ...]


def ensure_2d_dof(array: NDArray[T]) -> NDArray[T]:
    """Ensure an array is 2D, potentially adding a dof axis."""
    if array.ndim < 2:
        return array.reshape(-1, 1)
    assert array.ndim == 2
    return array


def ensure_2d_comp(array: NDArray[T]) -> NDArray[T]:
    """Ensure an array is 2D, potentially adding a comp axis."""
    if array.ndim < 2:
        return array.reshape(1, -1)
    assert array.ndim == 2
    return array


@define
class FieldData(Generic[T]):
    """Wrapper for a numpy array with a dof and a comp axis."""

    data: NDArray[T]

    def __attrs_post_init__(self) -> None:
        # Runtime assertion to ensure that only 2D arrays end up in here.
        assert self.data.ndim == 2

    @overload
    @staticmethod
    def join_comps(other: Iterable[FieldData[T] | NDArray[T]], /) -> FieldData[T]: ...

    @overload
    @staticmethod
    def join_comps(*other: FieldData[T] | NDArray[T]) -> FieldData[T]: ...

    @staticmethod
    def join_comps(*other):  # type: ignore[no-untyped-def]
        """Concatenate two or more arrays along the comp axis.

        Supports an arbitrary number of field data or numpy arrays, or a single
        iterable of such.
        """
        iterable = other if isinstance(other[0], FieldData | np.ndarray) else other[0]
        data = np.hstack([ensure_2d_dof(x.numpy() if isinstance(x, FieldData) else x) for x in iterable])
        return FieldData(data)

    @overload
    @staticmethod
    def join_dofs(other: Iterable[FieldData[T] | NDArray[T]], /) -> FieldData[T]: ...

    @overload
    @staticmethod
    def join_dofs(*other: FieldData[T] | NDArray[T]) -> FieldData[T]: ...

    @staticmethod
    def join_dofs(*other):  # type: ignore[no-untyped-def]
        """Join two or more arrays along the dof axis.

        Supports an arbitrary number field data or numpy arrays, or a single
        iterable of such.
        """
        iterable = other if isinstance(other[0], FieldData | np.ndarray) else other[0]
        data = np.vstack([ensure_2d_comp(x.numpy() if isinstance(x, FieldData) else x) for x in iterable])
        return FieldData(data)

    @overload
    @staticmethod
    def from_iter(
        iterable: Iterable[Iterable[float | int]], dtype: DTypeLike = float
    ) -> FieldData[floating]: ...

    @overload
    @staticmethod
    def from_iter(iterable: Iterable[Iterable[T]], dtype: DTypeLike = float) -> FieldData[T]: ...

    @staticmethod
    def from_iter(iterable, dtype: DTypeLike = float):  # type: ignore[no-untyped-def]
        """Construct a field data object from an iterable of iterables of scalars.
        The outer iterable loops through the rows (dof axis) and the inner
        iterables loop through the columns (comp axis).

        All methods that return a FieldData object either return `self` or
        creates a new one. For simplicity, always assume that the input object
        is destroyed when using such methods, and reassign to the output if
        necessary.
        """
        num_dofs = 0

        # Utility function for flattening the iterator. Keeps track of the
        # number of rows seen so that the final array can be reshaped in the
        # end.
        def values() -> Iterator:
            nonlocal num_dofs
            for value in iter(iterable):
                num_dofs += 1
                yield from value

        # This runs the values iterator to the end, which should populate the
        # local variable `ntuples`.
        array = np.fromiter(values(), dtype=dtype)

        return FieldData(array.reshape(num_dofs, -1))

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def num_comps(self) -> int:
        return self.data.shape[-1]

    @property
    def num_dofs(self) -> int:
        return self.data.shape[0]

    @property
    def comps(self) -> Iterable[NDArray[T]]:
        """Return a sequence of one-dimensional numpy arrays, one for each
        component.
        """
        return self.data.T

    @property
    def dofs(self) -> Iterable[NDArray[T]]:
        """Return a sequence of one-dimensional numpy arrays, one for each
        dof.
        """
        return self.data

    def mean(self) -> NDArray[T]:
        """Take the average over the dof axis."""
        return cast("NDArray[T]", self.data.mean(axis=0))

    def slice_comps(self, index: int | list[int]) -> FieldData[T]:
        """Extract a subset of components as a new field data object."""
        return FieldData(ensure_2d_comp(self.data[:, index]))

    def slice_dofs(self, index: int | list[int] | NDArray[integer]) -> FieldData[T]:
        """Extract a subset of dofs as a new field data object."""
        return FieldData(ensure_2d_dof(self.data[index, :]))

    def nan_filter(self, fill: T | None = None) -> FieldData[T]:
        """Fill NANs with a specified value. If set to None, will use the
        appropriate zero.
        """
        i = np.where(np.isnan(self.data))
        if len(i[0]) > 0:
            logging.warning(f"NaN values set to {fill}")
            self.data[i] = fill if fill is not None else np.zeros((), dtype=self.data.dtype)
        return self

    def ensure_ncomps(self, ncomps: int, allow_scalar: bool = False, pad_right: bool = True) -> FieldData[T]:
        """Ensure the data array has at least a certain number of components.

        Parameters:
        - ncomps: lower bound of number of components to allow.
        - allow_scalar: if true, allow exactly *one* component.
        - pad_right: if true (the default) fill zeros on the right (existing
            components will remain where they are)
        """
        if self.data.shape[-1] == 1 and allow_scalar:
            return self
        if self.data.shape[-1] >= ncomps:
            return self
        new_comps = ncomps - self.data.shape[-1]
        filler = np.zeros((self.data.shape[0], new_comps), dtype=self.data.dtype)
        to_stack = (self.data, filler) if pad_right else (filler, self.data)
        return FieldData(data=np.hstack(to_stack))

    def ensure_native(self) -> FieldData[T]:
        """Ensure the data array has native byte order."""
        if self.data.dtype.byteorder in ("=", sys.byteorder):
            return self
        swapped = self.data.byteswap()
        new_array = swapped.view(swapped.dtype.newbyteorder())
        return FieldData(new_array)

    def corners(self: FieldData[floating], shape: NodeShape) -> Points:
        """Return a sequence of corner points by interpreting the array as a
        cartesian product with a certain shape.
        """
        temp = self.data.reshape(*shape, -1)
        corners = temp[tuple(slice(None, None, j - 1) for j in temp.shape[:-1])]
        corners = corners.reshape(-1, self.num_comps)
        return Points(tuple(Point(tuple(corner)) for corner in corners))

    def collapse_weights(self: FieldData[floating]) -> FieldData[floating]:
        """Reduce the number of components by one, by dividing the first ncomps-1
        components with the last.

        This implements the NURBS normalization procedure, assuming the weights
        are stored as the last component.
        """
        data = self.data[..., :-1] / self.data[..., -1:]
        return FieldData(data)

    def transpose(self, shape: NodeShape, transposition: tuple[int, ...]) -> FieldData[T]:
        """Perform a transposition operation.

        Parameters:
        - shape: assumed shape of the contained data.
        - transposition: axial permutation.
        """
        return FieldData(
            self.data.reshape(*shape, -1)
            .transpose(*transposition, len(transposition))
            .reshape(self.data.shape)
        )

    def swap_components(self, i: int, j: int) -> FieldData[T]:
        """Swap two components by index."""
        self.data[:, i], self.data[:, j] = self.data[:, j].copy(), self.data[:, i].copy()
        return self

    def permute_components(self, permutation: Sequence[int]) -> FieldData[T]:
        return FieldData(self.data[:, permutation])

    @overload
    def constant_like(
        self, value: int, ndofs: int | None = None, ncomps: int | None = None, dtype: DTypeLike = None
    ) -> FieldData[integer]: ...

    @overload
    def constant_like(
        self, value: float, ndofs: int | None = None, ncomps: int | None = None, dtype: DTypeLike = None
    ) -> FieldData[floating]: ...

    def constant_like(self, value, ndofs=None, ncomps=None, dtype=None):  # type: ignore[no-untyped-def]
        """Return a new constant FieldData array.

        Parameters:
        - value: the value to fill the new array with.
        - ndofs: override the number of dofs.
        - ncomps: override the number of components.
        - dtype: override the data type.
        """
        retval = np.ones_like(
            self.data,
            shape=(
                ndofs or self.num_dofs,
                ncomps or self.num_comps,
            ),
            dtype=dtype or self.data.dtype,
        )
        retval.fill(value)
        return FieldData(retval)

    def trigonometric(self) -> FieldData[floating]:
        """Interpret the first two components as longitude and latitude, and
        return a new field data object with four components:

        - cos(longitude)
        - cos(latitude)
        - sin(longitude)
        - sin(latitude)
        """
        retval = np.zeros_like(self.data, shape=(self.num_dofs, 4))
        lon, lat, *_ = self.comps
        retval[:, 0] = np.cos(np.deg2rad(lon))
        retval[:, 1] = np.cos(np.deg2rad(lat))
        retval[:, 2] = np.sin(np.deg2rad(lon))
        retval[:, 3] = np.sin(np.deg2rad(lat))
        return FieldData(retval)

    def spherical_to_cartesian(self) -> FieldData[floating]:
        """Interpret the first two components as longitude and latitude, and
        return a new field data object with points in Cartesian coordinates. If
        there's a third component, it is interpreted as radius from the
        center.
        """
        clon, clat, slon, slat = self.trigonometric().comps
        retval = FieldData.join_comps(clon * clat, slon * clat, slat)
        if self.num_comps > 2:
            retval.data *= self.data[:, 2]
        return retval

    def cartesian_to_spherical(self, with_radius: bool = True) -> FieldData[floating]:
        """Interpret the components as x, y and z coordinates and return a new
        field data object with longitude and latitude.

        If with_radius is true, radius is included as the third component in the
        returned value.
        """
        x, y, z = self.comps
        lon = np.rad2deg(np.arctan2(y, x))
        lat = np.rad2deg(np.arctan(z / np.sqrt(x**2 + y**2)))

        if not with_radius:
            return FieldData.join_comps(lon, lat)

        radius = np.sqrt(x**2 + y**2 + z**2)
        return FieldData.join_comps(lon, lat, radius)

    def spherical_to_cartesian_vector_field(self, coords: FieldData[floating]) -> FieldData[floating]:
        """Interpret the components as a vector field in spherical coordinates
        (that is, longitudinal, latitudinal and radial components), and return a
        new field data object with the same vector field in Cartesian
        coordinates.

        Requires the coordinates of the corresponding points, in spherical
        coordinates, as input.
        """
        clon, clat, slon, slat = coords.trigonometric().comps
        u, v, w = self.comps
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
        """Interpret the components as a vector field in Cartesian coordinates,
        and return a new field data object with the same vector field in
        spherical coordinates (that is, longitudinal, latitudinal and radial
        components).

        Requires the coordinates of the corresponding points, in spherical
        coordinates, as input.
        """
        clon, clat, slon, slat = coords.trigonometric().comps
        u, v, w = self.comps
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
        """Apply a scipy rotation to the data and return a new field data
        object.
        """
        return FieldData(rotation.apply(self.data))

    def numpy(self, *shape: int) -> NDArray[T]:
        """Return the wrapped array as a numpy array, potentially reshaped."""
        if not shape:
            return self.data
        return self.data.reshape(*shape, self.num_comps)

    def vtk(self) -> vtkDataArray:
        """Return the wrapped array as a VTK array."""
        return cast("vtkDataArray", numpy_to_vtk(self.data, deep=1))

    def __add__(self, other: Any) -> FieldData:
        """Implement the '+' operator."""
        if isinstance(other, FieldData):
            return FieldData(self.data + other.data)
        return FieldData(self.data + other)

    @overload
    def __mul__(
        self: FieldData[integer], other: int | NDArray[integer] | FieldData[integer]
    ) -> FieldData[integer]: ...

    @overload
    def __mul__(
        self: FieldData[floating], other: int | float | NDArray[number] | FieldData[number]
    ) -> FieldData[floating]: ...

    def __mul__(self, other):  # type: ignore[no-untyped-def]
        """Implement the '*' operator."""
        if isinstance(other, FieldData):
            return FieldData(self.data * other.data)
        return FieldData(self.data * other)

    def __floordiv__(
        self: FieldData[integer], other: int | NDArray[integer] | FieldData[integer]
    ) -> FieldData[integer]:
        """Implement the '//' operator."""
        if isinstance(other, FieldData):
            return FieldData(self.data // other.data)
        return FieldData(self.data // other)

    def __truediv__(self, other: int | float | NDArray[number] | FieldData[number]) -> FieldData:
        """Implement the '/' operator."""
        if isinstance(other, FieldData):
            return FieldData(self.data / other.data)
        return FieldData(self.data / other)
