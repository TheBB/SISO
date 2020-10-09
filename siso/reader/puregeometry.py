from abc import ABC, abstractmethod
from pathlib import Path

import lrspline
from splipy.io import G2

from typing import Iterable, Tuple, Optional
from ..typing import StepData, Array2D

from .. import config, ConfigTarget
from ..geometry import SplineTopology, LRTopology, Patch
from ..fields import Field, SimpleField, Geometry, FieldPatches
from .reader import Reader



class PureGeometryReader(Reader, ABC):

    suffix: str
    filename: Path

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        return filename.suffix == cls.suffix

    def __init__(self, filename: Path):
        self.filename = filename

    def __enter__(self):
        self.f = open(self.filename).__enter__()
        return self

    def __exit__(self, *args):
        self.f.__exit__(*args)

    def validate(self):
        super().validate()
        config.require(multiple_timesteps=False, reason=f"{self.reader_name} do do not support multiple timesteps")
        config.ensure_limited(ConfigTarget.Reader, reason=f"not supported by {self.reader_name}")

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        yield (0, {'time': 0.0})

    def fields(self) -> Iterable[Field]:
        yield PureGeometryField(self)

    @abstractmethod
    def patches(self) -> FieldPatches:
        pass


class PureGeometryField(SimpleField):

    name = 'Geometry'
    fieldtype = Geometry()
    cells = False

    reader: PureGeometryReader

    def __init__(self, reader: PureGeometryReader):
        self.reader = reader

    def patches(self, stepid: int, force: bool = False, **_) -> FieldPatches:
        yield from self.reader.patches()


class G2Reader(PureGeometryReader):

    reader_name = "GoTools"
    suffix = '.g2'

    def patches(self):
        for i, (topo, data) in enumerate(SplineTopology.from_string(self.f.read())):
            yield Patch((i,), topo), data


class LRReader(PureGeometryReader):

    reader_name = "LRSplines"
    suffix = '.lr'

    def patches(self):
        for i, (topo, data) in enumerate(LRTopology.from_string(self.f.read())):
            yield Patch((i,), topo), data
