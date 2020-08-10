from pathlib import Path

from splipy.io import G2

from typing import List

from .. import config
from ..geometry import SplinePatch
from .reader import Reader


class G2Reader(Reader):

    reader_name = "GoTools"

    filename: Path
    patches: List[SplinePatch]

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        return filename.suffix == '.g2'

    def __init__(self, filename: Path):
        self.filename = filename
        config.require(multiple_timesteps=False)

    def __enter__(self):
        with G2(self.filename) as g2:
            self.patches = [
                SplinePatch((i,), patch) for i, patch
                in enumerate(g2.read())
            ]
        return self

    def __exit__(self, *args):
        pass

    def write(self, w):
        w.add_step(time=0.0)
        for patch in self.patches:
            w.update_geometry(patch)
        w.finalize_geometry()
        w.finalize_step()
