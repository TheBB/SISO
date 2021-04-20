from abc import abstractmethod
from pathlib import Path
import re
import sys

import f90nml
import numpy as np
from scipy.io import FortranFile, FortranFormattingError
import treelog as log

from typing import Optional, Iterable, Tuple, TextIO, Any, Dict
from ..typing import StepData, Array2D, Shape

from .reader import Reader
from .. import config, ConfigTarget
from ..coords import Local
from ..fields import Field, SimpleField, Geometry, FieldPatches
from ..geometry import Topology, UnstructuredTopology, StructuredTopology, Hex, Quad, Patch
from ..util import fortran_skip_record, save_excursion, cache, prod
from ..writer import Writer



# Utilities
# ----------------------------------------------------------------------


def dtypes(endianness):
    endian = {'native': '=', 'big': '>', 'small': '<'}[endianness]
    return np.dtype(f'{endian}f4'), np.dtype(f'{endian}u4')


def transpose(array, nodeshape):
    return array.reshape(*nodeshape, -1).transpose(1, 0, 2, 3).reshape(prod(nodeshape), -1)


def translate(path: Path, data: Array2D) -> Array2D:
    info_path = path / 'info.txt'
    if not info_path.exists():
        log.warning("Unable to find mesh origin info, coordinates may be unreliable")
        return data
    with open(info_path, 'r') as f:
        x, y = map(float, next(f).split())
    data[:,0] += x
    data[:,1] += y
    return data


def ensure_native(data: np.ndarray) -> np.ndarray:
    if data.dtype.byteorder in ('=', sys.byteorder):
        return data
    return data.byteswap().newbyteorder()



# Fields
# ----------------------------------------------------------------------


class SIMRAField(SimpleField):

    cells = False
    decompose = True

    index: int
    reader: 'SIMRAResultReader'
    scale: float

    def __init__(self, name: str, index: int, ncomps: int, reader: 'SIMRAResultReader', scale: float = 1.0):
        self.name = name
        self.index = index
        self.ncomps = ncomps
        self.reader = reader
        self.scale = scale

    def patches(self, stepid: int, force: bool = False, **_) -> FieldPatches:
        yield (
            self.reader.patch(),
            self.reader.data(stepid)[:, self.index : self.index + self.ncomps] * self.scale,
        )


class SIMRAGeometryField(SimpleField):

    name = 'Geometry'
    cells = False
    ncomps = 3

    reader: 'SIMRAReader'

    def __init__(self, reader: 'SIMRAReader'):
        self.reader = reader
        self.fieldtype = Geometry(Local().substitute())

    def patches(self, stepid: int, force: bool = False, **_) -> FieldPatches:
        yield self.reader.patch(), self.reader.nodes()



# Abstract reader
# ----------------------------------------------------------------------


class SIMRAReader(Reader):

    f4_type: np.dtype
    u4_type: np.dtype

    def validate(self):
        super().validate()
        config.ensure_limited(ConfigTarget.Reader, 'input_endianness', 'mesh_file', reason="not supported by SIMRA")

    def __init__(self):
        self.f4_type, self.u4_type = dtypes(config.input_endianness)

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        yield (0, {'time': 0.0})

    @abstractmethod
    def patch(self) -> Topology:
        pass

    @abstractmethod
    def nodes(self) -> Array2D:
        pass

    def fields(self) -> Iterable[Field]:
        yield SIMRAGeometryField(self)



# Concrete readers
# ----------------------------------------------------------------------


class SIMRAMeshReader(SIMRAReader):

    def validate(self):
        super().validate()
        config.require(multiple_timesteps=False, reason="SIMRA files do not support multiple timesteps")


class SIMRA2DMapReader(SIMRAMeshReader):

    reader_name = "SIMRA-map"

    filename: Path
    mapfile: TextIO
    shape: Tuple[int, int]

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        # The first line should have two right-aligned positive
        # integers, exactly eight characters each
        try:
            with open(filename, 'r') as f:
                line = next(f)
            assert len(line) == 17
            assert re.match(r'[ ]*[0-9]+', line[:8])
            assert re.match(r'[ ]*[0-9]+', line[8:16])
            assert line[-1] == '\n'
            return True
        except:
            return False

    def __init__(self, filename: Path):
        super().__init__()
        self.filename = filename

    def __enter__(self):
        self.mapfile = open(self.filename, 'r').__enter__()
        with save_excursion(self.mapfile):
            self.shape = tuple(int(s) - 1 for s in next(self.mapfile).split())
        return self

    def __exit__(self, *args):
        return self.mapfile.__exit__(*args)

    @property
    def nodeshape(self):
        return tuple(s+1 for s in self.shape)

    def patch(self):
        return Patch(('geometry',), StructuredTopology(self.shape[::-1], celltype=Quad()))

    def nodes(self):
        nodes = []
        with save_excursion(self.mapfile):
            next(self.mapfile)
            for line in self.mapfile:
                nodes.extend(map(float, line.split()))
        nodes = np.array(nodes).reshape(*self.nodeshape[::-1], 3)
        nodes[...,2 ] /= 10      # Map files have a vertical resolution factor of 10
        return translate(self.filename.parent, nodes.reshape(-1, 3))


class SIMRA2DMeshReader(SIMRAMeshReader):

    reader_name = "SIMRA-2D"

    filename: Path
    meshfile: TextIO
    shape: Tuple[int, int]

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        try:
            with open(filename, 'r') as f:
                assert next(f) == 'text\n'
            return True
        except:
            return False

    def __init__(self, filename: Path):
        super().__init__()
        self.filename = filename

    def __enter__(self):
        self.meshfile = open(self.filename, 'r').__enter__()
        next(self.meshfile)
        self.shape = tuple(int(s) - 1 for s in next(self.meshfile).split()[2:])
        return self

    def __exit__(self, *args):
        self.meshfile.__exit__(*args)

    @cache(1)
    def patch(self) -> Patch:
        return Patch(('geometry',), StructuredTopology(self.shape[::-1], celltype=Quad()))

    @cache(1)
    def nodes(self) -> Array2D:
        nnodes = prod(s+1 for s in self.shape)
        nodes = np.array([tuple(map(float, next(self.meshfile).split()[1:])) for _ in range(nnodes)])
        return translate(self.filename.parent, nodes)


class SIMRA3DMeshReader(SIMRAMeshReader):

    reader_name = "SIMRA-3D"

    filename: Path
    mesh: FortranFile
    nodeshape: Shape

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        _, u4_type = dtypes(config.input_endianness)
        try:
            with FortranFile(filename, 'r', header_dtype=u4_type) as f:
                assert f._read_size() == 6 * 4
            return True
        except:
            return False

    def __init__(self, filename: Path):
        super().__init__()
        self.filename = filename

    def __enter__(self):
        self.mesh = FortranFile(self.filename, 'r', header_dtype=self.u4_type).__enter__()
        with save_excursion(self.mesh._fp):
            _, _, imax, jmax, kmax, _ = self.mesh.read_ints(self.u4_type)
        self.nodeshape = (jmax, imax, kmax)
        return self

    def __exit__(self, *args):
        self.mesh.__exit__(*args)

    @cache(1)
    def patch(self) -> Patch:
        i, j, k = self.nodeshape
        return Patch(('geometry',), StructuredTopology((j-1, i-1, k-1), celltype=Hex()))

    @cache(1)
    def nodes(self) -> Array2D:
        with save_excursion(self.mesh._fp):
            fortran_skip_record(self.mesh)
            nodes = transpose(self.mesh.read_reals(self.f4_type), self.nodeshape)
        nodes = ensure_native(nodes)
        return translate(self.filename.parent, nodes)


class SIMRADataReader(SIMRAReader):

    result_fn: Path
    mesh_fn: Path
    input_fn: Path

    result: FortranFile
    mesh: SIMRA3DMeshReader
    input_data: Dict[str, Any]

    f4_type: np.dtype
    u4_type: np.dtype

    def __enter__(self):
        self.result = FortranFile(self.result_fn, 'r', header_dtype=self.u4_type).__enter__()
        self.mesh = SIMRA3DMeshReader(self.mesh_fn).__enter__()
        return self

    def __exit__(self, *args):
        self.mesh.__exit__(*args)
        self.result.__exit__(*args)

    def __init__(self, result_fn: Path):
        super().__init__()
        self.result_fn = Path(result_fn)
        self.mesh_fn = Path(config.mesh_file) if config.mesh_file else self.result_fn.with_name('mesh.dat')
        self.input_fn = self.result_fn.with_name('simra.in')

        if not self.mesh_fn.is_file():
            raise IOError(f"Unable to find mesh file: {self.mesh_fn}")

        if self.input_fn.is_file():
            self.input_data = f90nml.read(self.input_fn)
        else:
            self.input_data = {}
            log.warning(f"SIMRA input file not found, scales will be missing")

    def patch(self) -> Patch:
        return self.mesh.patch()

    def nodes(self) -> Array2D:
        return self.mesh.nodes()

    def scale(self, name: str) -> float:
        return self.input_data.get('param_data', {}).get(name, 1.0)

    def fields(self) -> Iterable[Field]:
        yield from self.mesh.fields()

        uref = self.scale('uref')
        lref = self.scale('lenref')
        yield SIMRAField('u', 0, 3, self, scale=uref)
        yield SIMRAField('ps', 3, 1, self, scale=uref**2)
        yield SIMRAField('tk', 4, 1, self, scale=uref**2)
        yield SIMRAField('td', 5, 1, self, scale=uref**3/lref)
        yield SIMRAField('vtef', 6, 1, self, scale=uref*lref)
        yield SIMRAField('pt', 7, 1, self)
        yield SIMRAField('pts', 8, 1, self)
        yield SIMRAField('rho', 9, 1, self)
        yield SIMRAField('rhos', 10, 1, self)

    @abstractmethod
    def data(self, stepid: int) -> Array2D:
        pass


class SIMRAContinuationReader(SIMRADataReader):

    reader_name = "SIMRA-Cont"

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        _, u4_type = dtypes(config.input_endianness)
        try:
            # It's too easy to mistake other files for SIMRA results,
            # so we require a certain suffix
            assert filename.suffix == '.res'
            with FortranFile(filename, 'r', header_dtype=u4_type) as f:
                size = f._read_size()
                assert size % u4_type.itemsize == 0
                assert size > u4_type.itemsize
                assert (size // u4_type.itemsize - 1) % 11 == 0  # Eleven scalars per point plus a time
            assert SIMRA3DMeshReader.applicable(
                Path(config.mesh_file) if config.mesh_file
                else filename.with_name('mesh.dat')
            )
            return True
        except:
            return False

    def validate(self):
        super().validate()
        config.require(multiple_timesteps=False, reason="SIMRA files do not support multiple timesteps")

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        # This is slightly hacky, but grabs the time value for the
        # next timestep without reading the whole dataset
        with save_excursion(self.result._fp):
            self.result._read_size()
            time = np.fromfile(self.result._fp, dtype=self.f4_type, count=1)[0]
        yield (0, {'time': time})

    @cache(1)
    def data(self, stepid: int) -> Array2D:
        data = self.result.read_reals(dtype=self.f4_type)
        _, data = data[0], data[1:]
        return ensure_native(transpose(data, self.mesh.nodeshape))


class SIMRAHistoryReader(SIMRADataReader):

    reader_name = "SIMRA-Hist"

    cur_stepid: int

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        _, u4_type = dtypes(config.input_endianness)
        try:
            # It's too easy to mistake other files for SIMRA results,
            # so we require a certain suffix
            assert filename.suffix == '.res'
            with FortranFile(filename, 'r', header_dtype=u4_type) as f:
                with save_excursion(f._fp):
                    size = f._read_size()
                    assert size == u4_type.itemsize
                assert f.read_ints(u4_type)[0] == u4_type.itemsize
                size = f._read_size()
                assert size % u4_type.itemsize == 0
                assert size > u4_type.itemsize
                assert (size // u4_type.itemsize - 1) % 12 == 0  # Twelve scalars per point plus a time
            assert SIMRA3DMeshReader.applicable(
                Path(config.mesh_file) if config.mesh_file
                else filename.with_name('mesh.dat')
            )
            return True
        except:
            return False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cur_stepid = -1

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        # Skip the word size indicator
        self.result.read_ints(self.u4_type)

        while True:
            # This is slightly hacky, but grabs the time value for the
            # next timestep without reading the whole dataset
            with save_excursion(self.result._fp):
                try:
                    self.result._read_size()
                except FortranFormattingError:
                    return
                time = np.fromfile(self.result._fp, dtype=self.f4_type, count=1)[0]

            self.cur_stepid += 1
            yield (self.cur_stepid, {'time': time})

    @cache(1)
    def data(self, stepid: int) -> Array2D:
        assert stepid == self.cur_stepid

        data = self.result.read_reals(dtype=self.f4_type)
        _, data = data[0], data[1:]

        # Skip the cell data
        self.result.read_reals(dtype=self.f4_type)

        return ensure_native(transpose(data, self.mesh.nodeshape))
