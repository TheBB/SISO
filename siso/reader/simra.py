from abc import abstractmethod
from pathlib import Path
import re

import numpy as np
from scipy.io import FortranFile

from typing import Optional, Iterable, Tuple, TextIO
from ..typing import StepData, Array2D

from .reader import Reader
from .. import config, ConfigTarget
from ..fields import Field, SimpleField, Geometry, FieldPatches
from ..geometry import Topology, UnstructuredTopology, StructuredTopology, Hex, Quad, Patch
from ..util import fortran_skip_record, save_excursion, cache
from ..writer import Writer



# Utilities
# ----------------------------------------------------------------------


def dtypes(endianness):
    endian = {'native': '=', 'big': '>', 'small': '<'}[endianness]
    return np.dtype(f'{endian}f4'), np.dtype(f'{endian}u4')



# Fields
# ----------------------------------------------------------------------


class SIMRAField(SimpleField):

    cells = False
    decompose = False

    index: int
    reader: 'SIMRAResultReader'

    def __init__(self, name: str, index: int, ncomps: int, reader: 'SIMRAResultReader'):
        self.name = name
        self.index = index
        self.ncomps = ncomps
        self.reader = reader

    def patches(self, stepid: int, force: bool = False, **_) -> FieldPatches:
        yield (
            self.reader.patch(),
            self.reader.data()[:, self.index : self.index + self.ncomps]
        )


class SIMRAGeometryField(SimpleField):

    name = 'Geometry'
    cells = False
    fieldtype = Geometry()
    ncomps = 3

    reader: 'SIMRAReader'

    def __init__(self, reader: 'SIMRAReader'):
        self.reader = reader

    def patches(self, stepid: int, force: bool = False, **_) -> FieldPatches:
        yield self.reader.patch(), self.reader.nodes()



# Abstract reader
# ----------------------------------------------------------------------


class SIMRAReader(Reader):

    f4_type: np.dtype
    u4_type: np.dtype

    def validate(self):
        super().validate()
        config.require(multiple_timesteps=False, reason="SIMRA files do not support multiple timesteps")
        config.ensure_limited(ConfigTarget.Reader, 'input_endianness', reason="not supported by SIMRA")

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


class SIMRA2DMeshReader(SIMRAReader):

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
        for line in self.mapfile:
            nodes.extend(map(float, line.split()))
        nodes = np.array(nodes).reshape(*self.nodeshape[::-1], 3)
        nodes[...,2 ] /= 10      # Map files have a vertical resolution factor of 10
        return nodes.reshape(-1, 3)


class SIMRA3DMeshReader(SIMRAReader):

    reader_name = "SIMRA-mesh"

    filename: Path
    mesh: FortranFile

    @classmethod
    def applicable(cls, filename:  Path) -> bool:
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
        return self

    def __exit__(self, *args):
        self.mesh.__exit__(*args)

    @cache(1)
    def patch(self) -> Patch:
        with save_excursion(self.mesh._fp):
            npts, nelems, _, _, _, _ = self.mesh.read_ints(self.u4_type)
            fortran_skip_record(self.mesh)
            cells = self.mesh.read_ints(self.u4_type).reshape(nelems, 8) - 1
            return Patch(('geometry',), UnstructuredTopology(npts, cells, celltype=Hex()))

    @cache(1)
    def nodes(self) -> Array2D:
        with save_excursion(self.mesh._fp):
            npts, _, _, _, _, _ = self.mesh.read_ints(self.u4_type)
            return self.mesh.read_reals(self.f4_type).reshape(npts, 3)


class SIMRAResultReader(SIMRAReader):

    reader_name = "SIMRA"

    result_fn: Path
    mesh_fn: Path

    result: FortranFile
    mesh: SIMRA3DMeshReader

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        _, u4_type = dtypes(config.input_endianness)
        try:
            # It's too easy to mistake other files for SIMRA results,
            # so we require a certain suffix
            assert filename.suffix == '.res'
            with FortranFile(filename, 'r', header_dtype=u4_type) as f:
                size = f._read_size()
                assert size % 4 == 0
                assert (size // 4 - 1) % 11 == 0  # Eleven scalars per point plus a time
            assert SIMRA3DMeshReader.applicable(filename.with_name('mesh.dat'))
            return True
        except:
            return False

    def __init__(self, result_fn: Path, mesh_fn: Optional[Path] = None):
        super().__init__()
        self.result_fn = Path(result_fn)
        self.mesh_fn = mesh_fn or self.result_fn.parent / 'mesh.dat'

    def __enter__(self):
        self.result = FortranFile(self.result_fn, 'r', header_dtype=self.u4_type).__enter__()
        self.mesh = SIMRA3DMeshReader(self.mesh_fn).__enter__()
        return self

    def __exit__(self, *args):
        self.mesh.__exit__(*args)
        self.result.__exit__(*args)

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        # This is slightly hacky, but grabs the time value for the
        # next timestep without reading the whole dataset
        with save_excursion(self.result._fp):
            self.result._read_size()
            time = np.fromfile(self.result._fp, dtype=self.f4_type, count=1)[0]
        yield (0, {'time': time})

    def patch(self) -> Patch:
        return self.mesh.patch()

    def nodes(self) -> Array2D:
        return self.mesh.nodes()

    @cache(1)
    def data(self) -> Array2D:
        data = self.result.read_reals(dtype=self.f4_type)
        _, data = data[0], data[1:].reshape(-1, 11)
        return data

    def fields(self) -> Iterable[Field]:
        yield from self.mesh.fields()
        yield SIMRAField('u', 0, 3, self)
        yield SIMRAField('ps', 3, 1, self)
        yield SIMRAField('tk', 4, 1, self)
        yield SIMRAField('td', 5, 1, self)
        yield SIMRAField('vtef', 6, 1, self)
        yield SIMRAField('pt', 7, 1, self)
        yield SIMRAField('pts', 8, 1, self)
        yield SIMRAField('rho', 9, 1, self)
        yield SIMRAField('rhos', 10, 1, self)
