from abc import ABC, abstractmethod
from pathlib import Path

import lrspline
from splipy.io import G2

from typing import Iterable

from .. import config
from ..geometry import Patch, SplinePatch, LRPatch
from .reader import Reader



class PureGeometryReader(Reader, ABC):

    suffix: str
    filename: Path

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        return filename.suffix == cls.suffix

    def __init__(self, filename: Path):
        self.filename = filename
        config.require(multiple_timesteps=False)

    @abstractmethod
    def patches(self) -> Iterable[Patch]:
        pass

    def write(self, w):
        w.add_step(time=0.0)
        for patch in self.patches():
            w.update_geometry(patch)
        w.finalize_geometry()
        w.finalize_step()


class G2Reader(PureGeometryReader):

    reader_name = "GoTools"
    suffix = '.g2'

    def __enter__(self):
        self.g2 = G2(str(self.filename)).__enter__()
        return self

    def __exit__(self, *args):
        self.g2.__exit__(*args)

    def patches(self):
        for i, patch in enumerate(self.g2.read()):
            yield SplinePatch((i,), patch)


class LRReader(PureGeometryReader):

    reader_name = "LRSplines"
    suffix = '.lr'

    def __enter__(self):
        self.lr = open(self.filename).__enter__()
        return self

    def __exit__(self, *args):
        self.lr.__exit__(*args)

    def patches(self):
        for i, patch in enumerate(lrspline.LRSplineObject.read_many(self.lr)):
            yield LRPatch((i,), patch)
