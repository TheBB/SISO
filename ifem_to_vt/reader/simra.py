from pathlib import Path

import numpy as np
from scipy.io import FortranFile

from typing import Optional, Iterable, Tuple
from ..typing import StepData, Array2D

from .. import config, ConfigTarget
from ..fields import Field, FieldPatch, SimpleFieldPatch
from ..geometry import Patch, UnstructuredPatch, Hex
from .reader import Reader
from ..writer import Writer
from ..util import save_excursion



class SIMRAField(Field):

    index: int
    reader: 'SIMRAReader'
    cells = False

    def __init__(self, name: str, index: int, ncomps: int, reader: 'SIMRAReader'):
        self.name = name
        self.index = index
        self.ncomps = ncomps
        self.decompose = False
        self.reader = reader

    def patches(self, stepid: int, force: bool = False) -> Iterable[FieldPatch]:
        yield SimpleFieldPatch(
            self.name, self.reader.patch(),
            self.reader.data()[:, self.index : self.index + self.ncomps]
        )



class SIMRAReader(Reader):

    reader_name = "SIMRA"

    result_fn: Path
    mesh_fn: Path

    result: FortranFile
    mesh: FortranFile

    _patch: Optional[Patch]
    _data: Optional[Array2D]

    @classmethod
    def applicable(self, filename: Path) -> bool:
        try:
            endian = {'native': '=', 'big': '>', 'small': '<'}[config.input_endianness]
            u4_type = f'{endian}u4'
            with FortranFile(filename, 'r', header_dtype=u4_type) as f:
                assert f._read_size() % 4 == 0
            with FortranFile(filename.with_name('mesh.dat'), 'r', header_dtype=u4_type) as f:
                assert f._read_size() == 6 * 4
            return True
        except:
            return False

    def __init__(self, result_fn: Path, mesh_fn: Optional[Path] = None):
        self.result_fn = Path(result_fn)
        self.mesh_fn = mesh_fn or self.result_fn.parent / 'mesh.dat'

        if not self.mesh_fn.is_file():
            raise IOError(f"Unable to find mesh file: {self.mesh_fn}")

        endian = {'native': '=', 'big': '>', 'small': '<'}[config.input_endianness]
        self.f4_type = np.dtype(f'{endian}f4')
        self.u4_type = np.dtype(f'{endian}u4')

        self._patch = None
        self._data = None

    def validate(self):
        super().validate()
        config.require(multiple_timesteps=False, reason="SIMRA files do not support multiple timesteps")
        config.ensure_limited(ConfigTarget.Reader, 'input_endianness', reason="not supported by SIMRA")

    def __enter__(self):
        self.result = FortranFile(self.result_fn, 'r', header_dtype=self.u4_type).__enter__()
        self.mesh = FortranFile(self.mesh_fn, 'r', header_dtype=self.u4_type).__enter__()
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
        if self._patch:
            return self._patch
        npts, nelems, imax, jmax, kmax, _ = self.mesh.read_ints(self.u4_type)
        coords = self.mesh.read_reals(self.f4_type).reshape(npts, 3)
        cells = self.mesh.read_ints(self.u4_type).reshape(nelems, 8) - 1
        patch = self._patch = UnstructuredPatch(('geometry',), coords, cells, celltype=Hex())
        return patch

    def data(self) -> Array2D:
        if self._data is not None:
            return self._data
        data = self.result.read_reals(dtype=self.f4_type)
        _, data = data[0], data[1:].reshape(-1, 11)
        self._data = data
        return data

    def geometry(self, stepid: int, force: bool = False) -> Iterable[Patch]:
        yield self.patch()

    def fields(self) -> Iterable[Field]:
        yield SIMRAField('u', 0, 3, self)
        yield SIMRAField('ps', 3, 1, self)
        yield SIMRAField('tk', 4, 1, self)
        yield SIMRAField('td', 5, 1, self)
        yield SIMRAField('vtef', 6, 1, self)
        yield SIMRAField('pt', 7, 1, self)
        yield SIMRAField('pts', 8, 1, self)
        yield SIMRAField('rho', 9, 1, self)
        yield SIMRAField('rhos', 10, 1, self)
