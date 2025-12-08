from __future__ import annotations

import logging
import re
from abc import abstractmethod
from contextlib import contextmanager
from enum import Enum, auto
from functools import lru_cache, partial
from itertools import count
from typing import (
    IO,
    TYPE_CHECKING,
    ClassVar,
    Literal,
    Self,
    TextIO,
    cast,
)

import f90nml
import numpy as np
import scipy.io
from attrs import define

from siso import api, util
from siso.api import (
    CellShape,
    NodeShape,
    Points,
    Zone,
    ZoneShape,
    impl_basis_of,
    impl_field_data,
    impl_fields,
    impl_geometries,
    impl_topology,
    impl_topology_updates,
    impl_use_geometry,
)
from siso.coord import Generic
from siso.impl import Basis, Field, Step
from siso.topology import CellType, StructuredTopology
from siso.util import FieldData

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from types import TracebackType

    from siso.types import Array, Matrix, Vector, f32d, f64d, u32d
    from siso.util.field_data import FloatFieldData

    from . import FindReaderSettings


class FortranFile(scipy.io.FortranFile):
    """Subclass of Scipy's Fortran file reader with some utility methods."""

    def skip_record(self) -> None:
        """Skip the current record and advance to the next one."""
        size = self._read_size()
        self._fp.seek(size, 1)
        assert self._read_size() == size

    def read_first[G: np.generic](self, dtype: np.dtype[G]) -> G:
        """Read only the first element from the current record, and advance to
        the next one.
        """
        size = self._read_size(eof_ok=True)
        retval = cast("G", np.fromfile(self._fp, dtype=dtype, count=1)[0])
        self._fp.seek(size - dtype.itemsize, 1)
        assert self._read_size() == size
        return retval

    def read_but_first[D: np.dtype](self, dtype: D) -> Vector[D]:
        """Read all but the first element from the current record, and advance
        to the next one.
        """
        size = self._read_size(eof_ok=True)
        self._fp.seek(dtype.itemsize, 1)
        retval = cast(
            "Vector[D]",
            np.fromfile(self._fp, dtype=dtype, count=size // dtype.itemsize - 1),
        )
        assert self._read_size() == size
        return retval

    def read_size(self) -> int:
        """Wrapper of `_read_size` to suppress warnings about private methods
        from linters.
        """
        return self._read_size()


# Type for enhancing Fortran files with random access. See `RandomAccessTracker`
# for more info. This uses FortranFile as the wrapper type and int as the marker
# type (each block numbered by index).
RandomAccessFortranTracker = util.RandomAccessTracker[FortranFile, int]


def fortran_marker_generator(tracker: RandomAccessFortranTracker) -> Iterator[tuple[int, int]]:
    """Marker generator for `RandomAccessFortranTracker`."""

    for i in count():
        # Use journey to keep track of the location
        with tracker.journey() as f:
            # Generate a mark for the current location
            marker = tracker.origin_marker(i)

            # Check if there's a block beginning at this location. If not,
            # abort.
            try:
                size = f._read_size(eof_ok=True)
            except scipy.io.FortranEOFError:
                return

            # Seek to the end of the block
            f._fp.seek(size, 1)
            assert f._read_size() == size

        # Having verified that a block exists, yield the marker. It's important
        # to do this outside the block that borrows the file pointer, since we
        # don't want to lay claim to it for too long.
        yield marker


class RandomAccessFortranFile(util.RandomAccessFile[FortranFile, int]):
    """Utility subclass for random access Fortran files."""

    def __init__(self, filename: Path, header_dtype: np.dtype):
        wrapper = cast(
            "Callable[[IO], FortranFile]",
            partial(FortranFile, header_dtype=header_dtype),
        )

        super().__init__(
            # No need to call __enter__ here, `RandomAccessFile` has an __enter__ method.
            fp=filename.open("rb"),
            # We intend to use FortranFile objects as file wrappers.
            wrapper=wrapper,
            # A marker generator that marks blocks by index.
            marker_generator=fortran_marker_generator,
        )


def transpose[D: np.dtype](array: Array[D], shape: tuple[int, ...]) -> Matrix[D]:
    """Utility function for transposing SIMRA arrays, which are generally
    ordered with a nasty mix of row-major and column-major ordering (z-axis
    varying the quickest, y the slowest and x in between.)
    """
    return array.reshape(*shape, -1).transpose(1, 0, 2, 3).reshape(util.prod(shape), -1)


def mesh_offset(root: Path, dim: Literal[2] | Literal[3]) -> Vector[f64d]:
    """Read the info.txt file in the same folder as the root path, and return a
    mesh offset, either a two- or three-element array.

    If the file does not exist, a warning is issued and zero is returned.
    """
    filename = root.parent / "info.txt"
    if not filename.exists():
        logging.warning("Unable to find mesh origin info (info.txt) - coordinates may be unreliable")
        return np.zeros((dim,), dtype=np.float64)
    with filename.open() as f:
        dx, dy = map(float, next(f).split())
    return np.array((dx, dy)) if dim == 2 else np.array((dx, dy, 0.0), dtype=np.float64)


def read_many[T: (int, float)](
    lines: Iterator[str], n: int, tp: Callable[[str], T], skip: bool = True
) -> np.ndarray:
    """Read n elements of type T from a sequence of lines, with potentially
    multiple elements per line, separated by space.

    Parameters:
    - lines: input data to read from
    - n: the number of elements to read
    - tp: callable for converting a string to a T
    - skip: if true, skip the first line
    """
    if skip:
        next(lines)
    values: list[T] = []
    while len(values) < n:
        values.extend(map(tp, next(lines).split()))
    return np.array(values)


def split_sparse(values: np.ndarray, ncomps: int = 1) -> tuple[np.ndarray, ...]:
    """Split the entries of an array by interpreting every nth element
    (beginning with the first one) as an index, and the intermediate elements as
    components (values to be placed at those indexes). Analogous to common
    sparse array storage schemes where the indexes and the values are stored
    interleaved. Returns the indexes and each component as a single-dimensional
    array in a tuple of length ncomps + 1.
    """
    jump = ncomps + 1
    indexes = values[::jump].astype(int)
    comps = [values[i::jump] for i in range(1, ncomps + 1)]
    return (indexes, *comps)


def make_mask(n: int, indices: np.ndarray, values: float | np.ndarray = 1.0) -> np.ndarray:
    """Return an array of length n, all zeros except placing `values` at
    `indices`. This is essentially just a converter from sparse to dense array
    formats.

    The values default to 1, which produces a on/off mask.
    """
    retval = np.zeros((n,), dtype=float)
    retval[indices - 1] = values
    return retval


@define
class SimraScales:
    """Class for reading scaling factors used for nondimensionalization.

    These are stored in a `simra.in` file in the same directory as the other
    files. If it doesn't exist, print a warning and proceed with unit scales.

    Parameters:
    - root: any path in the directory where `simra.in` is expected to be.
    """

    speed: float
    length: float

    @staticmethod
    def from_path(root: Path) -> SimraScales:
        filename = root.with_name("simra.in")

        params: dict[str, float]
        if filename.exists():
            params = f90nml.read(filename)["param_data"]
        else:
            logging.warning("Unable to find SIMRA input file (simra.in) - field scales may be unreliable")
            params = {}

        return SimraScales(
            speed=params.get("uref", 1.0),
            length=params.get("lenref", 1.0),
        )


class SimraMeshBase(api.Source[Basis, Field, Step, StructuredTopology, Zone[int]]):
    """Base class for all SIMRA mesh readers (map, 2D mesh and 3D mesh).

    Subclasses should populate the `simra_nodeshape` attribute on calling
    `__enter__`, set the `dim` classvar, as well as implement the `nodes`
    method.
    """

    filename: Path
    simra_nodeshape: NodeShape

    dim: ClassVar[int]

    def __init__(self, path: Path):
        self.filename = path

    @property
    def pardim(self) -> int:
        """Number of parametric dimensions."""
        return len(self.simra_nodeshape)

    @property
    def simra_cellshape(self) -> CellShape:
        return self.simra_nodeshape.cellular

    @property
    def out_nodeshape(self) -> NodeShape:
        """Convert SIMRA shapes to Siso shapes."""
        i, j, *rest = self.simra_nodeshape
        return NodeShape(j, i, *rest)

    @property
    def out_cellshape(self) -> CellShape:
        """Convert SIMRA shapes to Siso shapes."""
        return self.out_nodeshape.cellular

    @property
    def properties(self) -> api.SourceProperties:
        return api.SourceProperties(
            instantaneous=True,
            discrete_topology=True,
            single_basis=True,
            single_zoned=True,
            globally_keyed=True,
        )

    @abstractmethod
    def nodes(self) -> FloatFieldData:
        """Return the geometry nodal points."""
        ...

    def corners(self) -> Points:
        """Return the corners of the mesh."""
        return self.nodes().corners(self.simra_nodeshape)

    def configure(self, settings: api.ReaderSettings) -> None:
        """Override the filename if required."""
        if settings.mesh_filename:
            self.filename = settings.mesh_filename

    @impl_use_geometry
    def use_geometry(self, geometry: Field) -> None:
        return

    def bases(self) -> Iterator[Basis]:
        yield Basis("mesh")

    @impl_basis_of
    def basis_of(self, field: Field) -> Basis:
        return Basis("mesh")

    @impl_fields
    def fields(self, basis: Basis) -> Iterator[Field]:
        return
        yield

    @impl_geometries
    def geometries(self, basis: Basis) -> Iterator[Field]:
        yield Field("Geometry", type=api.Geometry(self.dim, coords=Generic()))

    def steps(self) -> Iterator[Step]:
        yield Step(index=0)

    def zones(self) -> Iterator[Zone[int]]:
        shape = ZoneShape.Hexahedron if self.pardim == 3 else ZoneShape.Quatrilateral
        yield Zone(shape=shape, coords=self.corners(), key=0)

    @impl_topology
    def topology(self, step: Step, basis: Basis, zone: Zone[int]) -> StructuredTopology:
        celltype = CellType.Hexahedron if self.pardim == 3 else CellType.Quadrilateral
        return StructuredTopology(self.out_cellshape, celltype, degree=1)

    @impl_field_data
    def field_data(self, timestep: Step, field: Field, zone: Zone[int]) -> FloatFieldData:
        return self.nodes()


class SimraMap(SimraMeshBase):
    """Reader for a SIMRA map file.

    A SIMRA map is a standard height map used as an input for the 2D mesh
    generator.
    """

    mesh: TextIO

    dim = 3

    @staticmethod
    def applicable(path: Path) -> bool:
        """Return true if `path` points to what is probably a SIMRA map file."""
        try:
            with path.open() as f:
                line = next(f)
            assert len(line) == 17
            assert re.match(r"[ ]*[0-9]*", line[:8])
            assert re.match(r"[ ]*[0-9]*", line[8:16])
            assert line[-1] == "\n"
            return True
        except (AssertionError, UnicodeDecodeError):
            return False

    def __enter__(self) -> Self:
        self.mesh = self.filename.open().__enter__()

        # Extract the shape without moving the file pointer
        with self.save_excursion():
            i, j = map(int, next(self.mesh).split())
        self.simra_nodeshape = NodeShape(i, j)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.mesh.__exit__(exc_type, exc_val, exc_tb)

    @contextmanager
    def save_excursion(self) -> Iterator[None]:
        """Context manager for saving and restoring a file pointer."""
        with util.save_excursion(self.mesh):
            yield

    def coords(self) -> Iterator[tuple[float, ...]]:
        """Consume the file and generate all the points in sequence."""
        with self.save_excursion():
            next(self.mesh)
            values: tuple[float, ...] = ()
            while True:
                if values:
                    point, values = values[:3], values[3:]
                    yield point
                    continue
                try:
                    line = next(self.mesh)
                except StopIteration:
                    return
                values = tuple(map(float, line.split()))

    @lru_cache(maxsize=1)
    def nodes(self) -> FieldData[f64d]:
        nodes = FieldData.from_iter(self.coords())

        # SIMRA map files have a vertical resolution of 10 cm.
        nodes.data[..., 2] /= 10

        return nodes + mesh_offset(self.filename, dim=3)


class Simra2dMesh(SimraMeshBase):
    """Reader for a SIMRA 2D mesh.

    A SIMRA 2D mesh is an exact slice of the surface level of the 3D mesh which
    is generated using the 2D mesh as an input.
    """

    mesh: TextIO

    dim = 2

    @staticmethod
    def applicable(path: Path) -> bool:
        """Return true if `path` points to what is probably a SIMRA 2D mesh
        file.
        """
        try:
            with path.open() as f:
                assert next(f) == "text\n"
            return True
        except (AssertionError, UnicodeDecodeError):
            return False

    def __enter__(self) -> Self:
        self.mesh = self.filename.open().__enter__()

        # Extract the shape without moving the file pointer
        with self.save_excursion():
            next(self.mesh)
            i, j = map(int, next(self.mesh).split()[2:])
        self.simra_nodeshape = NodeShape(i, j)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.mesh.__exit__(exc_type, exc_val, exc_tb)

    @contextmanager
    def save_excursion(self) -> Iterator[None]:
        """Context manager for saving and restoring a file pointer."""
        with util.save_excursion(self.mesh):
            yield

    @lru_cache(maxsize=1)
    def nodes(self) -> FieldData[f64d]:
        num_nodes = util.prod(self.simra_nodeshape)
        with self.save_excursion():
            next(self.mesh)
            next(self.mesh)

            # Iterate over the lines in the file, split on spaces and discard
            # the first element, then conver to float -> that's one node.
            nodes = FieldData.from_iter(map(float, next(self.mesh).split()[1:]) for _ in range(num_nodes))

        return nodes + mesh_offset(self.filename, dim=2)


class Simra3dMesh(SimraMeshBase):
    """Reader for a SIMRA 2D mesh.

    This class is used as a standalone reader, but also as an accessory to other
    SIMRA readers. See `SimraHasMesh` below.
    """

    mesh: RandomAccessFortranFile
    f4_type: np.dtype
    u4_type: np.dtype

    dim = 3

    @staticmethod
    def applicable(path: Path, settings: FindReaderSettings) -> bool:
        """Return true if `path` points to what is probably a SIMRA 3D mesh
        file.
        """
        u4_type = settings.endianness.u4_type()
        try:
            with FortranFile(path, "r", header_dtype=u4_type) as f:
                assert f._read_size() == 6 * 4
            return True
        except AssertionError:
            return False

    def configure(self, settings: api.ReaderSettings) -> None:
        super().configure(settings)
        self.f4_type = settings.endianness.f4_type()
        self.u4_type = settings.endianness.u4_type()

    def __enter__(self) -> Self:
        self.mesh = RandomAccessFortranFile(self.filename, header_dtype=self.u4_type).__enter__()

        # Leap to the first block (index zero) to get mesh metadata
        with self.mesh.leap(0) as f:
            _, _, imax, jmax, kmax, _ = f.read_ints(self.u4_type)
        self.simra_nodeshape = NodeShape(jmax, imax, kmax)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.mesh.__exit__(exc_type, exc_val, exc_tb)

    @lru_cache(maxsize=1)
    def nodes(self) -> FloatFieldData:
        # Leap to the second block (index one) to get the actual mesh coordinates
        with self.mesh.leap(1) as f:
            nodes = f.read_reals(self.f4_type)

        # Apply transposition to undo the stupid SIMRA node numbering, then
        # convert to native endianness.
        data: FloatFieldData = FieldData(transpose(nodes, self.simra_nodeshape)).ensure_native()

        return data + mesh_offset(self.filename, dim=3)


class SimraHasMesh(api.Source[Basis, Field, Step, StructuredTopology, Zone[int]]):
    """Base class for SIMRA readers that need access to a mesh. For all other
    SIMRA result files, the mesh is storead alongside with the file, not in the
    same file. This class provides the necessary implementation required to
    handle this.
    """

    mesh: Simra3dMesh

    @staticmethod
    def applicable(path: Path, settings: FindReaderSettings) -> bool:
        """Check whether an external mesh file is found."""
        return Simra3dMesh.applicable(settings.mesh_filename or path.with_name("mesh.dat"), settings)

    def __init__(self, path: Path, settings: FindReaderSettings):
        self.mesh = Simra3dMesh(settings.mesh_filename or path.with_name("mesh.dat"))

    def __enter__(self) -> Self:
        self.mesh.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.mesh.__exit__(exc_type, exc_val, exc_tb)

    def configure(self, settings: api.ReaderSettings) -> None:
        self.mesh.configure(settings)

    @impl_use_geometry
    def use_geometry(self, geometry: Field) -> None:
        return

    def zones(self) -> Iterator[Zone[int]]:
        return self.mesh.zones()

    def bases(self) -> Iterator[Basis]:
        return self.mesh.bases()

    @impl_basis_of
    def basis_of(self, field: Field) -> Basis:
        return next(self.mesh.bases())

    @impl_geometries
    def geometries(self, basis: Basis) -> Iterator[Field]:
        return self.mesh.geometries(basis)

    @impl_topology
    def topology(self, timestep: Step, basis: Basis, zone: Zone[int]) -> StructuredTopology:
        return self.mesh.topology(timestep, basis, zone)

    @impl_topology_updates
    def topology_updates(self, step: Step, basis: Basis) -> bool:
        return step.index == 0


class SimraBoundary(SimraHasMesh):
    """Reader for SIMRA boundary condition files.

    For each boundary condition field, this reader produces two fields: one for
    the actual values, and one mask, which has values of 1 on the points where
    the boundary conditions apply, and zero elsewhere.
    """

    filename: Path
    boundary: TextIO

    @staticmethod
    def applicable(path: Path, settings: FindReaderSettings) -> bool:
        """Check whether the path is probably a SIMRA boundary conditions
        file.
        """
        try:
            with path.open() as f:
                assert next(f).startswith("Boundary conditions")
            assert SimraHasMesh.applicable(path, settings)
            return True
        except (AssertionError, UnicodeDecodeError):
            return False

    def __init__(self, path: Path, settings: FindReaderSettings):
        super().__init__(path, settings)
        self.filename = path

    def __enter__(self) -> Self:
        super().__enter__()
        self.boundary = self.filename.open().__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)
        self.boundary.__exit__(exc_type, exc_val, exc_tb)

    @contextmanager
    def save_excursion(self) -> Iterator[None]:
        """Context manager for saving and restoring a file pointer."""
        with util.save_excursion(self.boundary):
            yield

    @property
    def properties(self) -> api.SourceProperties:
        # This reader produces one big composite field. Send split instructions
        # to the pipeline to split it apart.
        fields = [
            ("u", [0, 1, 2]),
            ("p", [3]),
            ("k", [4]),
            ("eps", [5]),
            ("pt", [6]),
            ("surface-roughness", [7]),
            ("u-mask", [8, 9, 10]),
            ("p-mask", [11]),
            ("wall-mask", [12]),
            ("log-mask", [13]),
            ("k,e-mask", [14]),
            ("pt-mask", [15]),
        ]
        splits = [
            api.SplitFieldSpec(
                source_name="nodal",
                new_name=name,
                components=components,
                splittable=True,
                destroy=True,
            )
            for name, components in fields
        ]

        return self.mesh.properties.update(
            split_fields=splits,
        )

    def steps(self) -> Iterator[Step]:
        yield Step(index=0)

    @impl_fields
    def fields(self, basis: Basis) -> Iterator[Field]:
        yield Field("nodal", type=api.Vector(16), splittable=False)

    @lru_cache(maxsize=1)
    def data(self) -> FieldData[f64d]:
        """Implementation of field_data for the non-geometry field."""
        with self.save_excursion():
            # Skip the first line
            next(self.boundary)

            # Number of fixed u, v, w, p, e, k values.
            *ints, _ = next(self.boundary).split()
            nfixu, nfixv, nfixw, nfixp, nfixe, nfixk, *rest, nlog = map(int, ints)

            # If there is anything between nfixk and nlog, it's nwalle, for
            # parallel SIMRA.
            is_parallel = bool(rest)
            nwalle: int | None = None
            if is_parallel:
                nwalle = rest[0]

            # Surface roughness: this field applies to the k=0 slice of the
            # mesh, so the ifix_z0 index array is not read from file.
            z0_var = read_many(self.boundary, nlog, float, skip=False)
            ifix_z0 = np.arange(1, util.prod(self.mesh.simra_nodeshape), self.mesh.simra_nodeshape[-1])

            # Velocity and pressure fields
            ifixu, fixu = split_sparse(read_many(self.boundary, 2 * nfixu, float))
            ifixv, fixv = split_sparse(read_many(self.boundary, 2 * nfixv, float))
            ifixw, fixw = split_sparse(read_many(self.boundary, 2 * nfixw, float))
            ifixp, fixp = split_sparse(read_many(self.boundary, 2 * nfixp, float))

            next(self.boundary)

            # If parallel, the walle field comes next. Ignore it.
            if is_parallel:
                assert nwalle is not None
                read_many(self.boundary, nwalle, int)

            # Wall mask and log lask
            t = read_many(self.boundary, 2 * nlog, int, skip=not is_parallel)
            iwall, ilog = t[::2], t[1::2]

            # k-epsilon mask and values
            ifixk = read_many(self.boundary, nfixk, int)
            t = read_many(self.boundary, 2 * nfixk, float, skip=False)
            fixk, fixd = t[::2], t[1::2]

            # If parallel, another field comes next. Ignore it too.
            npts = util.prod(self.mesh.simra_nodeshape)
            if is_parallel:
                read_many(self.boundary, npts, float, skip=False)

            # Temperature
            ifixtemp = read_many(self.boundary, nfixe, int)
            fixtemp = read_many(self.boundary, nfixe, float, skip=False)

        data = np.array(
            [
                make_mask(npts, ifixu, fixu),  # Velocity
                make_mask(npts, ifixv, fixv),
                make_mask(npts, ifixw, fixw),
                make_mask(npts, ifixp, fixp),  # Pressure
                make_mask(npts, ifixk, fixk),  # k-eps
                make_mask(npts, ifixk, fixd),
                make_mask(npts, ifixtemp, fixtemp),  # Temperature
                make_mask(npts, ifix_z0, z0_var),  # Surface roughness
                make_mask(npts, ifixu),  # Velocity mask
                make_mask(npts, ifixv),
                make_mask(npts, ifixw),
                make_mask(npts, ifixp),  # Pressure mask
                make_mask(npts, iwall),  # Wall mask
                make_mask(npts, ilog),  # Log mask
                make_mask(npts, ifixk),  # k-eps mask
                make_mask(npts, ifixtemp),  # Temperature mask
            ]
        ).T

        # Apply transposition to undo the stupid SIMRA node numbering.
        return FieldData(transpose(data, self.mesh.simra_nodeshape)).ensure_native()

    @impl_field_data
    def field_data(self, timestep: Step, field: Field, zone: Zone[int]) -> FloatFieldData:
        if field.is_geometry:
            return self.mesh.field_data(timestep, field, zone)
        return self.data()


class ExtraField(Enum):
    """Different possibilities for data in a continuation/initial condition file."""

    Nothing = auto()
    Stratification = auto()
    Pressure = auto()


class SimraContinuation(SimraHasMesh):
    """Reader for SIMRA continuation or initial condition files.

    There are a few variations of these files, similar enough that this reader
    can handle all of them.
    """

    filename: Path
    source: RandomAccessFortranFile

    # Possible variations of data that this reader can handle.
    extra_field: ExtraField = ExtraField.Nothing

    f4_type: np.dtype
    u4_type: np.dtype

    @staticmethod
    def applicable(path: Path, settings: FindReaderSettings) -> bool:
        """Check whether the path is probably a SIMRA continuation file or
        initial condition file.
        """
        u4_type = settings.endianness.u4_type()
        try:
            # Since these files have basically no identifying information, we
            # restrict ourselves to certain suffixes.
            assert path.suffix.casefold() in (".res", ".dat")

            with FortranFile(path, "r", header_dtype=u4_type) as f:
                size = f._read_size()
                assert size % u4_type.itemsize == 0
                assert size > u4_type.itemsize

                # Continuation files have 11 nodal fields and one timestep in
                # the first block
                if path.suffix.casefold() == ".res":
                    assert (size // u4_type.itemsize - 1) % 11 == 0
                # Initial condition files have 11 nodal fields
                elif path.suffix.casefold() == ".dat":
                    assert (size // u4_type.itemsize) % 11 == 0

            assert SimraHasMesh.applicable(path, settings)
            return True
        except AssertionError:
            return False

    def __init__(self, path: Path, settings: FindReaderSettings):
        super().__init__(path, settings)
        self.filename = path

    @property
    def is_init(self) -> bool:
        """Discriminate initial condition files from continuation files."""
        return self.filename.suffix.casefold() == ".dat"

    def configure(self, settings: api.ReaderSettings) -> None:
        super().configure(settings)
        self.f4_type = settings.endianness.f4_type()
        self.u4_type = settings.endianness.u4_type()

    @property
    def properties(self) -> api.SourceProperties:
        # This reader produces one big composite field. Send split instructions
        # to the pipeline to split it apart.
        fields = [
            ("u", [0, 1, 2]),
            ("ps", [3]),
            ("tk", [4]),
            ("td", [5]),
            ("vtef", [6]),
            ("pt", [7]),
            ("pts", [8]),
            ("rho", [9]),
            ("rhos", [10]),
        ]

        # Some files have an additional stratification field
        if self.extra_field == ExtraField.Stratification:
            fields.append(("strat", [11]))

        # Assemble the splits and return
        splits = [
            api.SplitFieldSpec(
                source_name="nodal",
                new_name=name,
                components=components,
                splittable=True,
                destroy=True,
            )
            for name, components in fields
        ]

        return self.mesh.properties.update(
            split_fields=splits,
        )

    def __enter__(self) -> Self:
        super().__enter__()
        self.source = RandomAccessFortranFile(self.filename, header_dtype=self.u4_type).__enter__()

        # Determine the exact form of the file by reading the size of the second block
        with self.source.leap(1) as f:
            try:
                size = f.read_size()
                # Nodal field: stratification
                if size == util.prod(self.mesh.out_nodeshape) * self.f4_type.itemsize:
                    self.extra_field = ExtraField.Stratification
                # Cellwise field: pressure?
                elif size == util.prod(self.mesh.out_cellshape) * self.f4_type.itemsize:
                    self.extra_field = ExtraField.Pressure
            # Exception: no extra field
            except scipy.io.FortranFormattingError:
                self.extra_field = ExtraField.Nothing
                pass
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)
        self.source.__exit__(exc_type, exc_val, exc_tb)

    @impl_fields
    def fields(self, basis: Basis) -> Iterator[Field]:
        # 11 or 12 nodal components, depending on whether the stratification
        # field is present.
        num_nodal = 11
        if self.extra_field == ExtraField.Stratification:
            num_nodal += 1
        yield Field("nodal", type=api.Vector(num_nodal), splittable=False)

        # Initial condition files have a cellwise pressure field
        if self.is_init:
            yield Field("pressure", type=api.Scalar(), cellwise=True)
        # Some continuation files also have a cellwise field. Is this pressure?
        elif self.extra_field == ExtraField.Pressure:
            yield Field("pressure?", type=api.Scalar(), cellwise=True)

    def steps(self) -> Iterator[Step]:
        with self.source.leap(0) as f:
            time = f.read_first(self.f4_type)
        yield Step(index=0, value=time)

    @lru_cache(maxsize=1)
    def data(self) -> tuple[FloatFieldData, FloatFieldData | None]:
        """Return a tuple of nodal data and optional cellwise data."""
        cells = None
        with self.source.leap(0) as f:
            # First block of nodal data: ignore the timestep value
            nodals = f.read_but_first(dtype=self.f4_type).reshape(-1, 11)
            if not self.is_init:
                # Continuation files with stratification field: add it to the
                # nodal array.
                if self.extra_field == ExtraField.Stratification:
                    extra = f.read_reals(dtype=self.f4_type)
                    nodals = np.hstack((nodals, extra.reshape(-1, 1)))
                # Contiunation files with pressure field: cellwise array
                elif self.extra_field == ExtraField.Pressure:
                    cells = f.read_reals(dtype=self.f4_type)
            # Initial condition files with pressure field: cellwise array
            else:
                cells = f.read_reals(dtype=self.f4_type)

        # Continuation file data must be scaled to physical units
        if not self.is_init:
            scales = SimraScales.from_path(self.filename)
            nodals[..., :3] *= scales.speed
            nodals[..., 3] *= scales.speed**2
            nodals[..., 4] *= scales.speed**2
            nodals[..., 5] *= scales.speed**3 / scales.length
            nodals[..., 6] *= scales.speed * scales.length

        # Construct field data objects, convert to native endianness and return.
        ndata: FloatFieldData = FieldData(transpose(nodals, self.mesh.simra_nodeshape)).ensure_native()

        cdata: FloatFieldData | None
        if cells is not None:
            cdata = FieldData(transpose(cells, self.mesh.simra_cellshape)).ensure_native()
        else:
            cdata = None

        return ndata, cdata

    @impl_field_data
    def field_data(self, timestep: Step, field: Field, zone: Zone[int]) -> FloatFieldData:
        if field.is_geometry:
            return self.mesh.field_data(timestep, field, zone)
        ndata, cdata = self.data()
        if field.cellwise:
            assert cdata is not None
            return cdata
        return ndata


class SimraHistory(SimraHasMesh):
    """Reader for SIMRA history files."""

    filename: Path
    source: RandomAccessFortranFile

    f4_type: f32d
    u4_type: u32d

    @staticmethod
    def applicable(path: Path, settings: FindReaderSettings) -> bool:
        """Check whether the path is probably a SIMRA history file."""
        u4_type = settings.endianness.u4_type()
        try:
            assert path.suffix.casefold() == ".res"
            with FortranFile(path, "r", header_dtype=u4_type) as f:
                # The first block should be a single integer equal to 4. This
                # serves as a nice 'magic number' as well as a check that the
                # endianness settings are correct.
                with util.save_excursion(f._fp):
                    size = f._read_size()
                    assert size == u4_type.itemsize
                assert f.read_ints(u4_type)[0] == u4_type.itemsize

                # The first block shouuld have a single time value and 12 nodal
                # fields.
                size = f._read_size()
                assert size % u4_type.itemsize == 0
                assert size > u4_type.itemsize
                assert (size // u4_type.itemsize - 1) % 12 == 0

            assert SimraHasMesh.applicable(path, settings)
            return True
        except AssertionError:
            return False

    def __init__(self, path: Path, settings: FindReaderSettings):
        super().__init__(path, settings)
        self.filename = path

    def configure(self, settings: api.ReaderSettings) -> None:
        super().configure(settings)
        self.f4_type = settings.endianness.f4_type()
        self.u4_type = settings.endianness.u4_type()

    @property
    def properties(self) -> api.SourceProperties:
        # This reader produces one big composite field. Send split instructions
        # to the pipeline to split it apart.
        fields = [
            ("u", [0, 1, 2]),
            ("ps", [3]),
            ("tk", [4]),
            ("td", [5]),
            ("vtef", [6]),
            ("pt", [7]),
            ("pts", [8]),
            ("rho", [9]),
            ("rhos", [10]),
        ]

        splits = [
            api.SplitFieldSpec(
                source_name="nodal",
                new_name=name,
                components=components,
                splittable=True,
                destroy=True,
            )
            for name, components in fields
        ]

        return self.mesh.properties.update(
            instantaneous=False,
            split_fields=splits,
        )

    def __enter__(self) -> Self:
        super().__enter__()
        self.source = RandomAccessFortranFile(self.filename, header_dtype=self.u4_type).__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)
        self.source.__exit__(exc_type, exc_val, exc_tb)

    @impl_fields
    def fields(self, basis: Basis) -> Iterator[Field]:
        yield Field("nodal", type=api.Vector(12))
        yield Field("pressure", type=api.Scalar(), cellwise=True)

    def steps(self) -> Iterator[Step]:
        # Every timestep constitutes two blocks in the file, starting with the
        # second one.
        for ts_index, rec_index in enumerate(count(start=1, step=2)):
            try:
                # The first float in the block is the time.
                with self.source.leap(rec_index) as f:
                    time = f.read_first(self.f4_type)
            except util.NoSuchMarkError:
                return
            yield Step(index=ts_index, value=time)

    @impl_field_data
    def field_data(self, timestep: Step, field: Field, zone: Zone[int]) -> FloatFieldData:
        if field.is_geometry:
            return self.mesh.field_data(timestep, field, zone)
        ndata, cdata = self.data(timestep.index)
        if field.cellwise:
            assert cdata is not None
            return cdata
        return ndata

    @lru_cache(maxsize=1)
    def data(self, index: int) -> tuple[FieldData[f32d], FieldData[f32d]]:
        """Return a tuple of nodal data and cellwise data."""

        # The first timestep uses block 1 and 2, the second uses blocks 3 and 4,
        # and so on. The first block is for nodal data (and time, which we skip),
        # and the second block for cellwise data.
        with self.source.leap(2 * index + 1) as f:
            ndata: Array[f32d] = f.read_but_first(self.f4_type)
            cdata: Array[f32d] = f.read_reals(self.f4_type)

        # Reshape, scale, construct field data objects, convert to native
        # endianness and return.
        ndata = ndata.reshape(-1, 12)
        scales = SimraScales.from_path(self.filename)
        ndata[..., :3] *= scales.speed
        ndata[..., 3] *= scales.speed**2
        ndata[..., 4] *= scales.speed**2
        ndata[..., 5] *= scales.speed**3 / scales.length
        ndata[..., 6] *= scales.speed * scales.length

        return (
            FieldData(transpose(ndata, self.mesh.simra_nodeshape)).ensure_native(),
            FieldData(transpose(cdata, self.mesh.simra_cellshape)).ensure_native(),
        )
