from abc import ABC, abstractmethod
from pathlib import Path

import lrspline
from splipy.io import G2

from typing import Iterable, Tuple
from ..typing import StepData

from .. import config, ConfigTarget
from ..geometry import Patch, SplinePatch, LRPatch
from ..fields import Field
from .reader import Reader



class PureGeometryReader(Reader, ABC):

    suffix: str
    filename: Path

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        return filename.suffix == cls.suffix

    def __init__(self, filename: Path):
        self.filename = filename

    def validate(self):
        super().validate()
        config.require(multiple_timesteps=False, reason=f"{self.reader_name} do do not support multiple timesteps")
        config.ensure_limited(ConfigTarget.Reader, reason=f"not supported by {self.reader_name}")

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        yield (0, {'time': 0.0})

    def fields(self) -> Iterable[Field]:
        return; yield


class G2Reader(PureGeometryReader):

    reader_name = "GoTools"
    suffix = '.g2'

    def __enter__(self):
        self.g2 = G2(str(self.filename)).__enter__()
        return self

    def __exit__(self, *args):
        self.g2.__exit__(*args)

    def geometry(self, stepid: int, force: bool = False):
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

    def geometry(self, stepid: int, force: bool = False):
        for i, patch in enumerate(lrspline.LRSplineObject.read_many(self.lr)):
            yield LRPatch((i,), patch)
