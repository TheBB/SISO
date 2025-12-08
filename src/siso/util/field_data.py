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
from itertools import product
from typing import TYPE_CHECKING, Self, cast, overload

import numpy as np
from numpy import integer
from numpy.typing import NDArray
from vtkmodules.util.numpy_support import numpy_to_vtk

from siso.api import NodeShape, Point, Points
from siso.types import (
    Array,
    Float,
    FloatArray,
    Floatd,
    FloatVector,
    Int,
    IntArray,
    Intd,
    IntVector,
    Matrix,
    Scalar,
    Vector,
    f32d,
    f64d,
    i32,
    i32d,
    i64,
    i64d,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from scipy.spatial.transform import Rotation
    from vtkmodules.vtkCommonCore import vtkDataArray


Index = int | slice | None | NDArray[integer]
Indices = Index | tuple[Index, ...]


def ensure_2d_dof[D: np.dtype](array: Array[D]) -> Matrix[D]:
    """Ensure an array is 2D, potentially adding a dof axis."""
    if array.ndim < 2:
        return array.reshape(-1, 1)
    assert array.ndim == 2
    return cast("Matrix[D]", array)


def ensure_2d_comp[D: np.dtype](array: Array[D]) -> Matrix[D]:
    """Ensure an array is 2D, potentially adding a comp axis."""
    if array.ndim < 2:
        return array.reshape(1, -1)
    assert array.ndim == 2
    return cast("Matrix[D]", array)


def pad_comps[D: np.dtype](array: Matrix[D], ncomps: int, value: Scalar) -> Matrix[D]:
    """Add extra components to an array."""
    if array.shape[1] == ncomps:
        return array
    return cast(
        "Matrix[D]",
        np.hstack([array, np.full((array.shape[0], ncomps - array.shape[1]), value, dtype=array.dtype)]),
    )


type IntFieldData = FieldData[i32d] | FieldData[i64d]
type FloatFieldData = FieldData[f32d] | FieldData[f64d]
type SliceParam = Int | Sequence[int] | Sequence[i32] | Sequence[i64] | IntVector


class FieldData[D: np.dtype]:
    """Wrapper for a numpy array with a dof and a comp axis."""

    data: Matrix[D]

    @overload
    def __init__[T: np.dtype](self, data: Array[T], /, *, dtype: D) -> None: ...

    @overload
    def __init__(self: FloatFieldData, data: FloatArray) -> None: ...

    @overload
    def __init__(self: IntFieldData, data: IntArray) -> None: ...

    def __init__(self, data, /, *, dtype=None):  # type: ignore[no-untyped-def]
        if dtype is not None:
            self.data = data.astype(dtype, casting="same_kind", copy=False)
        else:
            self.data = data
        assert self.data.ndim == 2

    @overload
    @staticmethod
    def join_comps(other: Iterable[FloatFieldData | FloatArray], /) -> FloatFieldData: ...

    @overload
    @staticmethod
    def join_comps(*other: FloatFieldData | FloatArray) -> FloatFieldData: ...

    @overload
    @staticmethod
    def join_comps(other: Iterable[IntFieldData | IntArray], /) -> IntFieldData: ...

    @overload
    @staticmethod
    def join_comps(*other: IntFieldData | IntArray) -> IntFieldData: ...

    @staticmethod
    def join_comps(*other):  # type: ignore[no-untyped-def]
        """Concatenate two or more arrays along the comp axis.

        Supports an arbitrary number of field data or numpy arrays, or a single
        iterable of such.
        """
        iterable = other if isinstance(other[0], FieldData | np.ndarray) else other[0]
        stack = [ensure_2d_dof(x.numpy() if isinstance(x, FieldData) else x) for x in iterable]
        data = np.hstack(stack)
        return FieldData(data)

    @overload
    @staticmethod
    def join_dofs(
        other: Iterable[FloatFieldData | FloatArray], /, *, pad_with: D | None = None
    ) -> FloatFieldData: ...

    @overload
    @staticmethod
    def join_dofs(*other: FloatFieldData | FloatArray, pad_with: D | None = None) -> FloatFieldData: ...

    @overload
    @staticmethod
    def join_dofs(other: Iterable[IntFieldData | IntArray], /) -> IntFieldData: ...

    @overload
    @staticmethod
    def join_dofs(*other: IntFieldData | IntArray) -> IntFieldData: ...

    @staticmethod
    def join_dofs(*other, pad_with=None):  # type: ignore[no-untyped-def]
        """Join two or more arrays along the dof axis.

        Supports an arbitrary number field data or numpy arrays, or a single
        iterable of such.
        """
        iterable = other if isinstance(other[0], FieldData | np.ndarray) else other[0]
        stack = [ensure_2d_comp(x.numpy() if isinstance(x, FieldData) else x) for x in iterable]
        if pad_with is not None:
            ncomps = max(x.shape[1] for x in stack)
            stack = [pad_comps(x, ncomps, pad_with) for x in stack]
        data = np.vstack(stack)
        return FieldData(data)

    @overload
    @staticmethod
    def from_iter(iterable: Iterable[Iterable[Scalar]]) -> FieldData[f64d]: ...

    @overload
    @staticmethod
    def from_iter[T: np.dtype](iterable: Iterable[Iterable[Scalar]], *, dtype: T) -> FieldData[T]: ...

    @overload
    @staticmethod
    def from_iter[T: np.generic](
        iterable: Iterable[Iterable[Scalar]], *, dtype: T
    ) -> FieldData[np.dtype[T]]: ...

    @staticmethod
    def from_iter(iterable, *, dtype=np.float64):  # type: ignore[no-untyped-def]
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
    def dtype(self) -> D:
        return self.data.dtype

    @property
    def num_comps(self) -> int:
        return self.data.shape[-1]

    @property
    def num_dofs(self) -> int:
        return self.data.shape[0]

    @property
    def comps(self) -> Iterable[Vector[D]]:
        """Return a sequence of one-dimensional numpy arrays, one for each
        component.
        """
        return cast("Iterable[Vector[D]]", self.data.T)

    @property
    def dofs(self) -> Iterable[Vector[D]]:
        """Return a sequence of one-dimensional numpy arrays, one for each
        dof.
        """
        return cast("Iterable[Vector[D]]", self.data)

    def mean(self) -> Vector[D]:
        """Take the average over the dof axis."""
        return cast("Vector[D]", self.data.mean(axis=0))

    def slice_comps(self, index: SliceParam) -> FieldData[D]:
        """Extract a subset of components as a new field data object."""
        return FieldData(ensure_2d_comp(self.data[:, index]))

    def slice_dofs(self, index: SliceParam) -> FieldData[D]:
        """Extract a subset of dofs as a new field data object."""
        return FieldData(ensure_2d_dof(self.data[index, :]))

    def nan_filter[G: Floatd](self: FieldData[G], fill: Scalar | None = None) -> FieldData[G]:
        """Fill NANs with a specified value. If set to None, will use the
        appropriate zero.
        """
        i = np.where(np.isnan(self.data))
        if len(i[0]) > 0:
            logging.warning(f"NaN values set to {fill}")
            self.data[i] = fill if fill is not None else np.zeros((), dtype=self.data.dtype)
        return self

    def ensure_ncomps(self, ncomps: int, allow_scalar: bool = False, pad_right: bool = True) -> FieldData[D]:
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
        data = cast("Matrix[D]", np.hstack(to_stack))
        return FieldData(data)

    def ensure_native(self) -> FieldData[D]:
        """Ensure the data array has native byte order."""
        if self.data.dtype.byteorder in ("=", sys.byteorder):
            return self
        swapped = self.data.byteswap()
        new_array = swapped.view(swapped.dtype.newbyteorder())
        return FieldData(new_array)

    def corners(self: FloatFieldData, shape: NodeShape) -> Points:
        """Return a sequence of corner points by interpreting the array as a
        cartesian product with a certain shape.
        """
        temp = self.data.reshape(*shape, -1)
        corners = temp[tuple(slice(None, None, j - 1) for j in temp.shape[:-1])]
        corners = corners.reshape(-1, self.num_comps)
        return Points(tuple(Point(tuple(corner)) for corner in corners))

    def bounding_corners(self: FloatFieldData) -> Points:
        """Return a bounding box as a sequence of corner points by interpreting
        the array as a point cloud.
        """
        minima: tuple[Float, ...] = tuple(np.min(comp) for comp in self.comps)  # type: ignore[misc]
        maxima: tuple[Float, ...] = tuple(np.max(comp) for comp in self.comps)  # type: ignore[misc]
        points = map(Point, product(*zip(minima, maxima)))
        return Points(tuple(points))

    def collapse_weights[G: Floatd](self: FieldData[G]) -> FieldData[G]:
        """Reduce the number of components by one, by dividing the first ncomps-1
        components with the last.

        This implements the NURBS normalization procedure, assuming the weights
        are stored as the last component.
        """
        data = self.data[..., :-1] / self.data[..., -1:]
        return FieldData(data)

    def transpose(self, shape: NodeShape, transposition: tuple[int, ...]) -> FieldData[D]:
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

    def swap_components(self, i: int, j: int) -> Self:
        """Swap two components by index."""
        self.data[:, i], self.data[:, j] = self.data[:, j].copy(), self.data[:, i].copy()
        return self

    def permute_components(self, permutation: Sequence[int]) -> FieldData[D]:
        return FieldData(self.data[:, permutation])

    @overload
    def constant_like(self, value: Int, *, ndofs: int | None = None, ncomps: int | None = None) -> Self: ...

    @overload
    def constant_like(
        self: FloatFieldData, value: Float, *, ndofs: int | None = None, ncomps: int | None = None
    ) -> FieldData[f64d]: ...

    @overload
    def constant_like[T: np.dtype](
        self, value: Scalar, *, dtype: T, ndofs: int | None = None, ncomps: int | None = None
    ) -> FieldData[T]: ...

    @overload
    def constant_like[T: np.generic](
        self, value: Scalar, *, dtype: type[T], ndofs: int | None = None, ncomps: int | None = None
    ) -> FieldData[np.dtype[T]]: ...

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

    def trigonometric(self) -> FloatFieldData:
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
        return cast("FloatFieldData", FieldData(retval))

    def spherical_to_cartesian(self: FloatFieldData) -> FloatFieldData:
        """Interpret the first two components as longitude and latitude, and
        return a new field data object with points in Cartesian coordinates. If
        there's a third component, it is interpreted as radius from the
        center.
        """
        clon, clat, slon, slat = self.trigonometric().comps
        retval = FieldData.join_comps(
            clon * clat,
            slon * clat,
            slat,
        )
        if self.num_comps > 2:
            retval *= self.data[:, 2]
        return retval

    def cartesian_to_spherical(self: FloatFieldData, with_radius: bool = True) -> FloatFieldData:
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

    def spherical_to_cartesian_vector_field(self: FloatFieldData, coords: FloatFieldData) -> FloatFieldData:
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

    def cartesian_to_spherical_vector_field(self: FloatFieldData, coords: FloatFieldData) -> FloatFieldData:
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

    def rotate(self: FloatFieldData, rotation: Rotation) -> FloatFieldData:
        """Apply a scipy rotation to the data and return a new field data
        object.
        """
        return cast("FloatFieldData", FieldData(rotation.apply(self.data)))

    @overload
    def numpy(self) -> Matrix[D]: ...

    @overload
    def numpy(self, s: int, /, *shape: int) -> Array[D]: ...

    def numpy(self, *shape: int) -> Array[D]:
        """Return the wrapped array as a numpy array, potentially reshaped."""
        if not shape:
            return self.data
        return self.data.reshape(*shape, self.num_comps)

    def vtk(self) -> vtkDataArray:
        """Return the wrapped array as a VTK array."""
        return cast("vtkDataArray", numpy_to_vtk(self.data, deep=1))

    @overload
    def __add__[T: Intd](self: FieldData[T], other: Int | IntVector | IntFieldData) -> FieldData[T]: ...

    @overload
    def __add__[T: Floatd](self: FieldData[T], other: Scalar | FloatVector | FieldData) -> FieldData[T]: ...

    def __add__(self, other) -> FieldData:  # type: ignore[no-untyped-def]
        """Implement the '+' operator."""
        if isinstance(other, FieldData):
            return FieldData(self.data + other.data)
        return FieldData(self.data + other)

    @overload
    def __mul__(self: IntFieldData, other: Int | IntArray | IntFieldData) -> IntFieldData: ...

    @overload
    def __mul__(self: FloatFieldData, other: Int | Float | FloatArray | FloatFieldData) -> FloatFieldData: ...

    def __mul__(self, other):  # type: ignore[no-untyped-def]
        """Implement the '*' operator."""
        if isinstance(other, FieldData):
            return FieldData(self.data * other.data)
        return FieldData(self.data * other)

    def __floordiv__(self: IntFieldData, other: Int | IntVector | IntFieldData) -> IntFieldData:
        """Implement the '//' operator."""
        if isinstance(other, FieldData):
            return FieldData(self.data // other.data)
        return cast("IntFieldData", FieldData(self.data // other))

    def __truediv__(
        self, other: Int | Float | FloatArray | IntArray | FloatFieldData | IntFieldData
    ) -> FloatFieldData:
        """Implement the '/' operator."""
        if isinstance(other, FieldData):
            return FieldData(self.data / other.data)
        return FieldData(self.data / other)
