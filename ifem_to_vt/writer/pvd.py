from os import makedirs
from os.path import splitext, join
from .vtu import Writer as AbstractVTUWriter

from .. import config


class Writer(AbstractVTUWriter):

    def __enter__(self):
        super().__enter__()
        self.pvd = open(self.filename, 'w')
        self.pvd.write('<VTKFile type="Collection">\n')
        self.pvd.write('  <Collection>\n')
        return self

    def __exit__(self, type_, value, backtrace):
        super().__exit__(type_, value, backtrace)
        self.pvd.write('  </Collection>\n')
        self.pvd.write('</VTKFile>\n')
        self.pvd.close()

    def make_filename(self):
        fn, ext = splitext(self.filename)
        root = join(fn, 'data')
        makedirs(root, mode=0o775, exist_ok=True)
        if not config.multiple_timesteps:
            return root + '.vtu'
        return '{}-{}.vtu'.format(root, self.stepid)

    def finalize_step(self):
        super().finalize_step()
        filename = self.make_filename()
        if self.step_data:
            timestep = next(iter(self.step_data.values()))
        else:
            timestep = self.stepid - 1
        self.pvd.write('    <DataSet timestep="{}" part="0" file="{}" />\n'.format(timestep, filename))
