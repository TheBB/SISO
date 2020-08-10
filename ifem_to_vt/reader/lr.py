from pathlib import Path

import lrspline

from typing import List

from .. import config
from ..geometry import LRPatch
from .reader import Reader


class LRReader(Reader):

    reader_name = "LRSplines"

    filename: Path
    patches: List[LRPatch]

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        return filename.suffix == '.lr'

    def __init__(self, filename: Path):
        self.filename = filename
        config.require(multiple_timesteps=False)

    def __enter__(self):
        with open(self.filename, 'rb') as f:
            self.patches = [
                LRPatch((i,), patch) for i, patch
                in enumerate(lrspline.LRSplineObject.read_many(f))
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
