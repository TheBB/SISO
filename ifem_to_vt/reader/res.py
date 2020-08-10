from pathlib import Path

from scipy.io import FortranFile

from .. import config
from ..fields import SimpleFieldPatch
from ..geometry import UnstructuredPatch, Hex
from .reader import Reader


class SIMRAReader(Reader):

    reader_name = "SIMRA"

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

    def __init__(self, result_fn, mesh_fn=None):
        config.require(multiple_timesteps=False)
        self.result_fn = Path(result_fn)
        if mesh_fn is None:
            self.mesh_fn = self.result_fn.parent / 'mesh.dat'
        else:
            self.mesh_fn = mesh_fn

        if not self.mesh_fn.is_file():
            raise IOError('Unable to find mesh file: {}'.format(self.mesh_fn))

        endian = {'native': '=', 'big': '>', 'small': '<'}[config.input_endianness]
        self.f4_type = endian + 'f4'
        self.u4_type = endian + 'u4'

    def __enter__(self):
        self.result = FortranFile(self.result_fn, 'r', header_dtype=self.u4_type).__enter__()
        self.mesh = FortranFile(self.mesh_fn, 'r', header_dtype=self.u4_type).__enter__()
        return self

    def __exit__(self, type_, value, backtrace):
        self.mesh.__exit__(type_, value, backtrace)
        self.result.__exit__(type_, value, backtrace)

    def write(self, w):
        npts, nelems, imax, jmax, kmax, _ = self.mesh.read_ints(self.u4_type)
        coords = self.mesh.read_reals(self.f4_type).reshape(npts, 3)
        cells = self.mesh.read_ints(self.u4_type).reshape(nelems, 8) - 1
        patch = UnstructuredPatch(('geometry',), coords, cells, celltype=Hex())

        data = self.result.read_reals(dtype=self.f4_type)
        time, data = data[0], data[1:].reshape(-1, 11)

        w.add_step(time=time)
        w.update_geometry(patch)
        w.finalize_geometry()

        w.update_field(SimpleFieldPatch('u',    patch, data[:,:3]))
        w.update_field(SimpleFieldPatch('ps',   patch, data[:,3:4]))
        w.update_field(SimpleFieldPatch('tk',   patch, data[:,4:5]))
        w.update_field(SimpleFieldPatch('td',   patch, data[:,5:6]))
        w.update_field(SimpleFieldPatch('vtef', patch, data[:,6:7]))
        w.update_field(SimpleFieldPatch('pt',   patch, data[:,7:8]))
        w.update_field(SimpleFieldPatch('pts',  patch, data[:,8:9]))
        w.update_field(SimpleFieldPatch('rho',  patch, data[:,9:10]))
        w.update_field(SimpleFieldPatch('rhos', patch, data[:,10:11]))

        w.finalize_step()
