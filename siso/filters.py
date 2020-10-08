from abc import ABC, abstractmethod
from contextlib import contextmanager

from .fields import Field, PatchData, FieldData

from .typing import StepData, Array2D
from typing import ContextManager



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
