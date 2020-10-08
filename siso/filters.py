from abc import ABC, abstractmethod
from contextlib import contextmanager

from . import config
from .fields import Field, PatchData, FieldData

from .typing import StepData, Array2D
from typing import ContextManager, Iterable, Tuple



# Abstract base classes
# ----------------------------------------------------------------------


class Sink(ABC):

    @abstractmethod
    def step(self, stepdata: StepData) -> ContextManager['StepFilter']:
        pass


class StepSink(ABC):

    @abstractmethod
    def geometry(self, field: Field) -> ContextManager['FieldFilter']:
        pass

    @abstractmethod
    def field(self, field: Field) -> ContextManager['FieldFilter']:
        pass


class FieldSink(ABC):

    @abstractmethod
    def __call__(self, patch: PatchData, data: FieldData):
        pass


class Source(ABC):

    @abstractmethod
    def steps(self) -> Iterable[Tuple[int, StepData]]:
        """Iterate over all steps with associated data."""
        pass

    @abstractmethod
    def fields(self) -> Iterable[Field]:
        """Iterate over all fields."""
        pass



# LastStep
# ----------------------------------------------------------------------


class LastStepFilter(Source):

    src: Source

    def __init__(self, src: Source):
        """Filter that only returns the last timestep."""
        self.src = src
        config.require(multiple_timesteps=False)

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        for step in self.src.steps():
            pass
        yield step

    def fields(self) -> Iterable[Field]:
        yield from self.src.fields()
