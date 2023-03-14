from __future__ import annotations

import logging
import re
from abc import abstractmethod
from contextlib import contextmanager
from enum import Enum, auto
from functools import lru_cache, partial
from itertools import count
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    TextIO,
    Tuple,
    TypeVar,
    Union,
)

import f90nml
import numpy as np
import scipy.io
from attrs import define
from numpy import floating
from typing_extensions import Self

from .. import api, util
from ..coord import Generic
from ..field import Field
from ..timestep import Step
from ..topology import CellType, StructuredTopology, Topology
from ..util import FieldData
from ..zone import Coords, Shape, Zone
from . import FindReaderSettings


class FortranFile(scipy.io.FortranFile):
    def jump(self):
        return util.FileJump(self._fp)

    def skip_record(self):
        size = self._read_size()
        self._fp.seek(size, 1)
        assert self._read_size() == size

    def read_first(self, dtype: np.dtype) -> Any:
        size = self._read_size(eof_ok=True)
        retval = np.fromfile(self._fp, dtype=dtype, count=1)[0]
        self._fp.seek(size - dtype.itemsize, 1)
        assert self._read_size() == size
        return retval

    def read_but_first(self, dtype: np.dtype) -> np.ndarray:
        size = self._read_size(eof_ok=True)
        self._fp.seek(dtype.itemsize, 1)
        retval = np.fromfile(self._fp, dtype=dtype, count=size // dtype.itemsize - 1)
        assert self._read_size() == size
        return retval

    def read_size(self) -> int:
        return self._read_size()


RandomAccessFortranTracker = util.RandomAccessTracker[FortranFile, int]


def fortran_marker_generator(tracker: RandomAccessFortranTracker) -> Iterator[Tuple[int, int]]:
    for i in count():
        with tracker.journey() as f:
            marker = tracker.origin_marker(i)
            try:
                size = f._read_size(eof_ok=True)
            except scipy.io.FortranEOFError:
                return
            f._fp.seek(size, 1)
            assert f._read_size() == size
        yield marker


class RandomAccessFortranFile(util.RandomAccessFile[FortranFile, int]):
    def __init__(self, filename: Path, header_dtype: np.dtype):
        super().__init__(
            fp=open(filename, "rb"),
            wrapper=partial(FortranFile, header_dtype=header_dtype),
            marker_generator=fortran_marker_generator,
        )


def transpose(array: np.ndarray, shape: Tuple[int, ...]):
    return array.reshape(*shape, -1).transpose(1, 0, 2, 3).reshape(util.prod(shape), -1)


def mesh_offset(root: Path, dim: Union[Literal[2], Literal[3]]) -> np.ndarray:
    filename = root.parent / "info.txt"
    if not filename.exists():
        logging.warning("Unable to find mesh origin info (info.txt) - coordinates may be unreliable")
        return np.zeros((dim,), dtype=np.float32)
    else:
        with open(filename, "r") as f:
            dx, dy = map(float, next(f).split())
        return np.array((dx, dy)) if dim == 2 else np.array((dx, dy, 0.0))


T = TypeVar("T", int, float)


def read_many(lines: Iterator[str], n: int, tp: Callable[[str], T], skip: bool = True) -> np.ndarray:
    if skip:
        next(lines)
    values: List[T] = []
    while len(values) < n:
        values.extend(map(tp, next(lines).split()))
    return np.array(values)


def split_sparse(values: np.ndarray, ncomps: int = 1) -> Tuple[np.ndarray, ...]:
    jump = ncomps + 1
    indexes = values[::jump].astype(int)
    comps = [values[i::jump] for i in range(1, ncomps + 1)]
    return (indexes, *comps)


def make_mask(n: int, indices: np.ndarray, values: Union[float, np.ndarray] = 1.0) -> np.ndarray:
    retval = np.zeros((n,), dtype=float)
    retval[indices - 1] = values
    return retval


@define
class SimraScales:
    speed: float
    length: float

    @staticmethod
    def from_path(root: Path) -> SimraScales:
        filename = root.with_name("simra.in")

        params: Dict[str, float]
        if filename.exists():
            params = f90nml.read(filename)["param_data"]
        else:
            logging.warning("Unable to find SIMRA input file (simra.in) - field scales may be unreliable")
            params = {}

        return SimraScales(
            speed=params.get("uref", 1.0),
            length=params.get("lenref", 1.0),
        )


class SimraMeshBase(api.Source[Field, Step, Zone]):
    filename: Path
    simra_nodeshape: Tuple[int, ...]

    dim: ClassVar[int]

    def __init__(self, path: Path):
        self.filename = path

    @property
    def pardim(self) -> int:
        return len(self.simra_nodeshape)

    @property
    def simra_cellshape(self) -> Tuple[int, ...]:
        return tuple(i - 1 for i in self.simra_nodeshape)

    @property
    def out_nodeshape(self) -> Tuple[int, ...]:
        i, j, *rest = self.simra_nodeshape
        return (j, i, *rest)

    @property
    def out_cellshape(self) -> Tuple[int, ...]:
        return tuple(i - 1 for i in self.out_nodeshape)

    @property
    def properties(self) -> api.SourceProperties:
        return api.SourceProperties(
            instantaneous=True,
        )

    @abstractmethod
    def nodes(self) -> FieldData[floating]:
        ...

    def corners(self) -> Coords:
        return self.nodes().corners(self.simra_nodeshape)

    def configure(self, settings: api.ReaderSettings) -> None:
        if settings.mesh_filename:
            self.filename = settings.mesh_filename

    def use_geometry(self, geometry: Field) -> None:
        return

    def fields(self) -> Iterator[Field]:
        yield Field("Geometry", type=api.Geometry(self.dim, coords=Generic()))

    def steps(self) -> Iterator[Step]:
        yield Step(index=0)

    def zones(self) -> Iterator[Zone]:
        shape = Shape.Hexahedron if self.pardim == 3 else Shape.Quatrilateral
        yield Zone(
            shape=shape,
            coords=self.corners(),
            local_key="0",
        )

    def topology(self, timestep: Step, field: Field, zone: Zone) -> StructuredTopology:
        celltype = CellType.Hexahedron if self.pardim == 3 else CellType.Quadrilateral
        return StructuredTopology(self.out_cellshape, celltype)

    def field_data(self, timestep: Step, field: Field, zone: Zone) -> FieldData[floating]:
        return self.nodes()


class SimraMap(SimraMeshBase):
    mesh: TextIO

    dim = 3

    @staticmethod
    def applicable(path: Path) -> bool:
        try:
            with open(path, "r") as f:
                line = next(f)
            assert len(line) == 17
            assert re.match(r"[ ]*[0-9]*", line[:8])
            assert re.match(r"[ ]*[0-9]*", line[8:16])
            assert line[-1] == "\n"
            return True
        except (AssertionError, UnicodeDecodeError):
            return False

    def __enter__(self) -> Self:
        self.mesh = open(self.filename, "r").__enter__()
        with self.save_excursion():
            i, j = map(int, next(self.mesh).split())
        self.simra_nodeshape = (i, j)
        return self

    def __exit__(self, *args):
        self.mesh.__exit__(*args)

    @contextmanager
    def save_excursion(self):
        with util.save_excursion(self.mesh):
            yield

    def coords(self) -> Iterator[Tuple[float, ...]]:
        with self.save_excursion():
            next(self.mesh)
            values: Tuple[float, ...] = ()
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
    def nodes(self) -> FieldData[floating]:
        nodes = FieldData.from_iter(self.coords())
        nodes.data[..., 2] /= 10
        return nodes + mesh_offset(self.filename, dim=3)


class Simra2dMesh(SimraMeshBase):
    mesh: TextIO

    dim = 2

    @staticmethod
    def applicable(path: Path) -> bool:
        try:
            with open(path, "r") as f:
                assert next(f) == "text\n"
            return True
        except (AssertionError, UnicodeDecodeError):
            return False

    def __enter__(self) -> Self:
        self.mesh = open(self.filename, "r").__enter__()
        next(self.mesh)
        i, j = map(int, next(self.mesh).split()[2:])
        self.simra_nodeshape = (i, j)
        return self

    def __exit__(self, *args):
        self.mesh.__exit__(*args)

    @contextmanager
    def save_excursion(self):
        with util.save_excursion(self.mesh):
            yield

    @property
    def out_cellshape(self) -> Tuple[int, int]:
        i, j = self.out_nodeshape
        return (i - 1, j - 1)

    @lru_cache(maxsize=1)
    def nodes(self) -> FieldData[floating]:
        num_nodes = util.prod(self.simra_nodeshape)
        nodes = FieldData.from_iter(map(float, next(self.mesh).split()[1:]) for _ in range(num_nodes))
        return nodes + mesh_offset(self.filename, dim=2)


class Simra3dMesh(SimraMeshBase):
    mesh: RandomAccessFortranFile
    f4_type: np.dtype
    u4_type: np.dtype

    dim = 3

    @staticmethod
    def applicable(path: Path, settings: FindReaderSettings) -> bool:
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
        with self.mesh.leap(0) as f:
            _, _, imax, jmax, kmax, _ = f.read_ints(self.u4_type)
        self.simra_nodeshape = (jmax, imax, kmax)
        return self

    def __exit__(self, *args) -> None:
        self.mesh.__exit__(*args)

    @lru_cache(maxsize=1)
    def nodes(self) -> FieldData[floating]:
        with self.mesh.leap(1) as f:
            nodes = f.read_reals(self.f4_type)
        data = FieldData(transpose(nodes, self.simra_nodeshape)).ensure_native()
        return data + mesh_offset(self.filename, dim=3)


class SimraBoundary(api.Source[Field, Step, Zone]):
    filename: Path
    boundary: TextIO

    @staticmethod
    def applicable(path: Path, settings: FindReaderSettings) -> bool:
        try:
            with open(path, "r") as f:
                assert next(f).startswith("Boundary conditions")
            assert Simra3dMesh.applicable(settings.mesh_filename or path.with_name("mesh.dat"), settings)
            return True
        except (AssertionError, UnicodeDecodeError):
            return False

    def __init__(self, path: Path):
        self.filename = path
        self.mesh = Simra3dMesh(path.with_name("mesh.dat"))

    def __enter__(self) -> Self:
        self.boundary = open(self.filename, "r").__enter__()
        self.mesh.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self.boundary.__exit__(*args)
        self.mesh.__exit__(*args)

    @contextmanager
    def save_excursion(self):
        with util.save_excursion(self.boundary):
            yield

    def configure(self, settings: api.ReaderSettings) -> None:
        self.mesh.configure(settings)

    @property
    def properties(self) -> api.SourceProperties:
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
        return api.SourceProperties(
            instantaneous=True,
            split_fields=splits,
        )

    def use_geometry(self, geometry: Field) -> None:
        return

    def steps(self) -> Iterator[Step]:
        yield Step(index=0)

    def zones(self) -> Iterator[Zone]:
        yield from self.mesh.zones()

    def fields(self) -> Iterator[Field]:
        yield from self.mesh.fields()
        yield Field("nodal", type=api.Vector(16), splittable=False)

    def topology(self, timestep: Step, field: Field, zone: Zone) -> Topology:
        return self.mesh.topology(timestep, field, zone)

    @lru_cache(maxsize=1)
    def data(self) -> FieldData[floating]:
        with self.save_excursion():
            next(self.boundary)

            *ints, _ = next(self.boundary).split()
            nfixu, nfixv, nfixw, nfixp, nfixe, nfixk, *rest, nlog = map(int, ints)

            is_parallel = bool(rest)
            nwalle: Optional[int] = None
            if is_parallel:
                nwalle = rest[0]

            z0_var = read_many(self.boundary, nlog, float, skip=False)
            ifix_z0 = np.arange(1, util.prod(self.mesh.simra_nodeshape), self.mesh.simra_nodeshape[-1])

            ifixu, fixu = split_sparse(read_many(self.boundary, 2 * nfixu, float))
            ifixv, fixv = split_sparse(read_many(self.boundary, 2 * nfixv, float))
            ifixw, fixw = split_sparse(read_many(self.boundary, 2 * nfixw, float))
            ifixp, fixp = split_sparse(read_many(self.boundary, 2 * nfixp, float))

            next(self.boundary)

            if is_parallel:
                assert nwalle is not None
                read_many(self.boundary, nwalle, int)

            t = read_many(self.boundary, 2 * nlog, int, skip=not is_parallel)
            iwall, ilog = t[::2], t[1::2]

            ifixk = read_many(self.boundary, nfixk, int)
            t = read_many(self.boundary, 2 * nfixk, float, skip=False)
            fixk, fixd = t[::2], t[1::2]

            npts = util.prod(self.mesh.simra_nodeshape)
            if is_parallel:
                read_many(self.boundary, npts, float, skip=False)

            ifixtemp = read_many(self.boundary, nfixe, int)
            fixtemp = read_many(self.boundary, nfixe, float, skip=False)

        data = np.array(
            [
                make_mask(npts, ifixu, fixu),
                make_mask(npts, ifixv, fixv),
                make_mask(npts, ifixw, fixw),
                make_mask(npts, ifixp, fixp),
                make_mask(npts, ifixk, fixk),
                make_mask(npts, ifixk, fixd),
                make_mask(npts, ifixtemp, fixtemp),
                make_mask(npts, ifix_z0, z0_var),
                make_mask(npts, ifixu),
                make_mask(npts, ifixv),
                make_mask(npts, ifixw),
                make_mask(npts, ifixp),
                make_mask(npts, iwall),
                make_mask(npts, ilog),
                make_mask(npts, ifixk),
                make_mask(npts, ifixtemp),
            ]
        ).T

        return FieldData(transpose(data, self.mesh.simra_nodeshape)).ensure_native()

    def field_data(self, timestep: Step, field: Field, zone: Zone) -> FieldData[floating]:
        if field.is_geometry:
            return self.mesh.field_data(timestep, field, zone)
        return self.data()


class ExtraField(Enum):
    Nothing = auto()
    Stratification = auto()
    Unknown = auto()


class SimraContinuation(api.Source[Field, Step, Zone]):
    filename: Path
    source: RandomAccessFortranFile
    mesh: Simra3dMesh

    extra_field: ExtraField = ExtraField.Nothing

    f4_type: np.dtype
    u4_type: np.dtype

    @staticmethod
    def applicable(path: Path, settings: FindReaderSettings) -> bool:
        u4_type = settings.endianness.u4_type()
        try:
            assert path.suffix.casefold() in (".res", ".dat")
            with FortranFile(path, "r", header_dtype=u4_type) as f:
                size = f._read_size()
                assert size % u4_type.itemsize == 0
                assert size > u4_type.itemsize
                if path.suffix.casefold() == ".res":
                    assert (size // u4_type.itemsize - 1) % 11 == 0
                elif path.suffix.casefold() == ".dat":
                    assert (size // u4_type.itemsize) % 11 == 0
            assert Simra3dMesh.applicable(settings.mesh_filename or path.with_name("mesh.dat"), settings)
            return True
        except AssertionError:
            return False

    def __init__(self, path: Path):
        self.filename = path
        self.mesh = Simra3dMesh(path.with_name("mesh.dat"))

    @property
    def is_init(self) -> bool:
        return self.filename.suffix.casefold() == ".dat"

    def configure(self, settings: api.ReaderSettings) -> None:
        self.f4_type = settings.endianness.f4_type()
        self.u4_type = settings.endianness.u4_type()
        self.mesh.configure(settings)

    @property
    def properties(self) -> api.SourceProperties:
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
        if self.extra_field == ExtraField.Stratification:
            fields.append(("strat", [11]))
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
        return api.SourceProperties(
            instantaneous=True,
            split_fields=splits,
        )

    def __enter__(self) -> Self:
        self.source = RandomAccessFortranFile(self.filename, header_dtype=self.u4_type).__enter__()
        self.mesh.__enter__()

        with self.source.leap(1) as f:
            try:
                size = f.read_size()
                if size == util.prod(self.mesh.out_nodeshape) * self.f4_type.itemsize:
                    self.extra_field = ExtraField.Stratification
                elif size == util.prod(self.mesh.out_cellshape) * self.f4_type.itemsize:
                    self.extra_field = ExtraField.Unknown
            except scipy.io.FortranFormattingError:
                pass
        return self

    def __exit__(self, *args) -> None:
        self.source.__exit__(*args)
        self.mesh.__exit__(*args)

    def use_geometry(self, geometry: Field) -> None:
        return

    def fields(self) -> Iterator[Field]:
        yield from self.mesh.fields()

        num_nodal = 11
        if self.extra_field == ExtraField.Stratification:
            num_nodal += 1
        yield Field("nodal", type=api.Vector(num_nodal), splittable=False)

        if self.is_init:
            yield Field("pressure", type=api.Scalar(), cellwise=True)
        elif self.extra_field == ExtraField.Unknown:
            yield Field("pressure?", type=api.Scalar(), cellwise=True)

    def steps(self) -> Iterator[Step]:
        with self.source.leap(0) as f:
            time = f.read_first(self.f4_type)
        yield Step(index=0, value=time)

    def zones(self) -> Iterator[Zone]:
        yield from self.mesh.zones()

    def topology(self, timestep: Step, field: Field, zone: Zone) -> Topology:
        return self.mesh.topology(timestep, field, zone)

    @lru_cache(maxsize=1)
    def data(self) -> Tuple[FieldData[floating], Optional[FieldData[floating]]]:
        cells = None
        with self.source.leap(0) as f:
            nodals = f.read_but_first(dtype=self.f4_type)
            if not self.is_init:
                nodals = nodals.reshape(-1, 11)
                if self.extra_field == ExtraField.Stratification:
                    extra = f.read_reals(dtype=self.f4_type)
                    nodals = np.hstack((nodals, extra.reshape(-1, 1)))
                elif self.extra_field == ExtraField.Unknown:
                    cells = f.read_reals(dtype=self.f4_type)
            else:
                cells = f.read_reals(dtype=self.f4_type)

        if not self.is_init:
            scales = SimraScales.from_path(self.filename)
            nodals[..., :3] *= scales.speed
            nodals[..., 3] *= scales.speed**2
            nodals[..., 4] *= scales.speed**2
            nodals[..., 5] *= scales.speed**3 / scales.length
            nodals[..., 6] *= scales.speed * scales.length

        ndata = FieldData(transpose(nodals, self.mesh.simra_nodeshape)).ensure_native()
        if cells is not None:
            cdata = FieldData(transpose(cells, self.mesh.simra_cellshape)).ensure_native()
        else:
            cdata = None
        return ndata, cdata

    def field_data(self, timestep: Step, field: Field, zone: Zone) -> FieldData[floating]:
        if field.is_geometry:
            return self.mesh.field_data(timestep, field, zone)
        ndata, cdata = self.data()
        if field.cellwise:
            assert cdata is not None
            return cdata
        return ndata


class SimraHistory(api.Source[Field, Step, Zone]):
    filename: Path
    source: RandomAccessFortranFile
    mesh: Simra3dMesh

    f4_type: np.dtype
    u4_type: np.dtype

    @staticmethod
    def applicable(path: Path, settings: FindReaderSettings) -> bool:
        u4_type = settings.endianness.u4_type()
        try:
            assert path.suffix.casefold() == ".res"
            with FortranFile(path, "r", header_dtype=u4_type) as f:
                with util.save_excursion(f._fp):
                    size = f._read_size()
                    assert size == u4_type.itemsize
                assert f.read_ints(u4_type)[0] == u4_type.itemsize
                size = f._read_size()
                assert size % u4_type.itemsize == 0
                assert size > u4_type.itemsize
                assert (size // u4_type.itemsize - 1) % 12 == 0
            assert Simra3dMesh.applicable(settings.mesh_filename or path.with_name("mesh.dat"), settings)
            return True
        except AssertionError:
            return False

    def __init__(self, path: Path):
        self.filename = path
        self.mesh = Simra3dMesh(path.with_name("mesh.dat"))

    def configure(self, settings: api.ReaderSettings) -> None:
        self.f4_type = settings.endianness.f4_type()
        self.u4_type = settings.endianness.u4_type()
        self.mesh.configure(settings)

    @property
    def properties(self) -> api.SourceProperties:
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
        return api.SourceProperties(
            split_fields=splits,
            instantaneous=False,
        )

    def __enter__(self) -> Self:
        self.source = RandomAccessFortranFile(self.filename, header_dtype=self.u4_type).__enter__()
        self.mesh.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self.source.__exit__(*args)
        self.mesh.__exit__(*args)

    def use_geometry(self, geometry: Field) -> None:
        return

    def fields(self) -> Iterator[Field]:
        yield from self.mesh.fields()
        yield Field("nodal", type=api.Vector(12), splittable=False)
        yield Field("pressure", type=api.Scalar(), cellwise=True)

    def steps(self) -> Iterator[Step]:
        for ts_index, rec_index in enumerate(count(start=1, step=2)):
            try:
                with self.source.leap(rec_index) as f:
                    time = f.read_first(self.f4_type)
            except util.NoSuchMarkError:
                return
            yield Step(index=ts_index, value=time)

    def zones(self) -> Iterator[Zone]:
        yield from self.mesh.zones()

    def topology(self, timestep: Step, field: Field, zone: Zone) -> Topology:
        return self.mesh.topology(timestep, field, zone)

    def field_data(self, timestep: Step, field: Field, zone: Zone) -> FieldData[floating]:
        if field.is_geometry:
            return self.mesh.field_data(timestep, field, zone)
        ndata, cdata = self.data(timestep.index)
        if field.cellwise:
            assert cdata is not None
            return cdata
        return ndata

    @lru_cache(maxsize=1)
    def data(self, index: int) -> Tuple[FieldData[floating], FieldData[floating]]:
        with self.source.leap(2 * index + 1) as f:
            ndata = f.read_but_first(self.f4_type)
            cdata = f.read_reals(self.f4_type)

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
