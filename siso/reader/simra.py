from pathlib import Path

import f90nml
import numpy as np
from scipy.io import FortranFile
import treelog as log

from typing import Optional, Iterable, Tuple
from ..typing import StepData, Array2D

from .reader import Reader
from .. import config, ConfigTarget
from ..fields import Field, SimpleField, Geometry, FieldPatches
from ..geometry import Patch, UnstructuredPatch, StructuredPatch, Hex
from ..util import fortran_skip_record, save_excursion, cache
from ..writer import Writer



class SIMRAField(SimpleField):

    cells = False
    decompose = True

    index: int
    reader: 'SIMRAReader'
    scale: float

    def __init__(self, name: str, index: int, ncomps: int, reader: 'SIMRAReader', scale: float = 1.0):
        self.name = name
        self.index = index
        self.ncomps = ncomps
        self.reader = reader
        self.scale = scale

    def patches(self, stepid: int, force: bool = False, **_) -> FieldPatches:
        yield (
            self.reader.patch(),
            self.reader.data()[:, self.index : self.index + self.ncomps] * self.scale,
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



class SIMRAReader(Reader):

    reader_name = "SIMRA"

    result_fn: Path
    mesh_fn: Path

    result: FortranFile
    mesh: FortranFile

    f4_type: np.dtype
    u4_type: np.dtype

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

    def __init__(self, result_fn: Path, mesh_fn: Optional[Path] = None, input_fn: Optional[Path] = None):
        self.result_fn = Path(result_fn)
        self.mesh_fn = mesh_fn or self.result_fn.parent / 'mesh.dat'
        self.input_fn = input_fn or self.result_fn.parent / 'simra.in'

        if not self.mesh_fn.is_file():
            raise IOError(f"Unable to find mesh file: {self.mesh_fn}")

        if self.input_fn.is_file():
            self.input_data = f90nml.read(self.input_fn)
        else:
            self.input_data = {}
            log.warning(f"SIMRA input file not found, scales will be missing")

        endian = {'native': '=', 'big': '>', 'small': '<'}[config.input_endianness]
        self.f4_type = np.dtype(f'{endian}f4')
        self.u4_type = np.dtype(f'{endian}u4')

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

    @cache(1)
    def patch(self) -> Patch:
        with save_excursion(self.mesh._fp):
            npts, nelems, imax, jmax, kmax, _ = self.mesh.read_ints(self.u4_type)
            fortran_skip_record(self.mesh)
            cells = self.mesh.read_ints(self.u4_type).reshape(nelems, 8) - 1
            return StructuredPatch(('geometry',), (imax-1, jmax-1, kmax-1), celltype=Hex())

    @cache(1)
    def nodes(self) -> Array2D:
        with save_excursion(self.mesh._fp):
            npts, _, _, _, _, _ = self.mesh.read_ints(self.u4_type)
            return self.mesh.read_reals(self.f4_type).reshape(npts, 3)

    @cache(1)
    def data(self) -> Array2D:
        data = self.result.read_reals(dtype=self.f4_type)
        _, data = data[0], data[1:].reshape(-1, 11)
        return data

    def scale(self, name: str) -> float:
        return self.input_data.get('param_data', {}).get(name, 1.0)

    def fields(self) -> Iterable[Field]:
        uref = self.scale('uref')
        lref = self.scale('lenref')

        yield SIMRAGeometryField(self)
        yield SIMRAField('u', 0, 3, self, scale=uref)
        yield SIMRAField('ps', 3, 1, self, scale=uref**2)
        yield SIMRAField('tk', 4, 1, self, scale=uref**2)
        yield SIMRAField('td', 5, 1, self, scale=uref**3/lref)
        yield SIMRAField('vtef', 6, 1, self, scale=uref*lref)
        yield SIMRAField('pt', 7, 1, self)
        yield SIMRAField('pts', 8, 1, self)
        yield SIMRAField('rho', 9, 1, self)
        yield SIMRAField('rhos', 10, 1, self)
