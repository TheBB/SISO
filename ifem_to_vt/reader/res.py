from pathlib import Path

from scipy.io import FortranFile

from .. import config


class Reader:

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
        elements = self.mesh.read_ints(self.u4_type).reshape(nelems, 8) - 1

        data = self.result.read_reals(dtype=self.f4_type)
        time, data = data[0], data[1:].reshape(-1, 11)

        w.add_step(time=time)
        w.update_geometry(coords, elements, 3, 0)
        w.finalize_geometry(0)

        w.update_field(data[:,:3], 'u',    0, 0, 'vector', cells=False)
        w.update_field(data[:,3],  'ps',   0, 0, 'scalar', cells=False)
        w.update_field(data[:,4],  'tk',   0, 0, 'scalar', cells=False)
        w.update_field(data[:,5],  'td',   0, 0, 'scalar', cells=False)
        w.update_field(data[:,6],  'vtef', 0, 0, 'scalar', cells=False)
        w.update_field(data[:,7],  'pt',   0, 0, 'scalar', cells=False)
        w.update_field(data[:,8],  'pts',  0, 0, 'scalar', cells=False)
        w.update_field(data[:,9],  'rho',  0, 0, 'scalar', cells=False)
        w.update_field(data[:,10], 'rhos', 0, 0, 'scalar', cells=False)

        w.finalize_step()
