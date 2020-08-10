from pathlib import Path
from os import makedirs

from .vtu import VTUWriter
from .. import config


class PVDWriter(VTUWriter):

    writer_name = "PVD"

    @classmethod
    def applicable(self, fmt: str) -> bool:
        return fmt == 'pvd'

    def __init__(self, outpath: Path):
        outpath = Path(outpath)
        self.rootfile = outpath
        super().__init__(outpath.with_suffix('') / 'data.vtu')

    def __enter__(self):
        super().__enter__()
        self.pvd = open(self.rootfile, 'w')
        self.pvd.write('<VTKFile type="Collection">\n')
        self.pvd.write('  <Collection>\n')
        return self

    def __exit__(self, type_, value, backtrace):
        super().__exit__(type_, value, backtrace)
        self.pvd.write('  </Collection>\n')
        self.pvd.write('</VTKFile>\n')
        self.pvd.close()

    def make_filename(self, *args, **kwargs):
        filename = super().make_filename(*args, **kwargs)
        makedirs(filename.parent, mode=0o775, exist_ok=True)
        return filename

    def finalize_step(self):
        super().finalize_step()
        filename = self.make_filename(with_step=True)
        if self.stepdata:
            timestep = next(iter(self.stepdata.values()))
        else:
            timestep = self.stepid
        self.pvd.write('    <DataSet timestep="{}" part="0" file="{}" />\n'.format(timestep, filename))
