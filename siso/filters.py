from abc import ABC, abstractmethod
from contextlib import contextmanager

from .fields import Field, PatchData, FieldData

from .typing import StepData, Array2D
from typing import ContextManager



# Abstract base classes
# ----------------------------------------------------------------------


class Filter(ABC):

    @abstractmethod
    def step(self, stepdata: StepData) -> ContextManager['StepFilter']:
        pass


class StepFilter(ABC):

    @abstractmethod
    def geometry(self) -> ContextManager['FieldFilter']:
        pass

    @abstractmethod
    def field(self) -> ContextManager['FieldFilter']:
        pass


class FieldFilter(ABC):

    @abstractmethod
    def __call__(self, field: Field, patch: PatchData, data: FieldData):
        pass
