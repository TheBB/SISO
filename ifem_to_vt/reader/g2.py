from pathlib import Path

from splipy.io import G2

from .. import config
from ..geometry import SplinePatch
from .hdf5 import GeometryManager
from .reader import Reader


class SimpleBasis:

    def __init__(self, patches):
        self.patches = patches
        self.name = 'Geometry'

    @property
    def npatches(self):
        return len(self.patches)

    def update_at(self, stepid):
        return stepid == 0

    def patch_at(self, stepid, patchid):
        return self.patches[patchid]


class G2Reader(Reader):

    reader_name = "GoTools"

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        return filename.suffix == '.g2'

    def __init__(self, filename):
        self.filename = filename
        config.require(multiple_timesteps=False)

    def __enter__(self):
        with G2(self.filename) as g2:
            patches = [
                SplinePatch((i,), patch) for i, patch
                in enumerate(g2.read())
            ]
        self.basis = SimpleBasis(patches)
        return self

    def __exit__(self, type_, value, backtrace):
        pass

    def write(self, w):
        w.add_step(time=0.0)
        geometry = GeometryManager(self.basis)
        geometry.update(w, 0)
        w.finalize_step()
