from abc import ABC, abstractmethod

import numpy as np
import treelog as log

from typing import List, Optional, Iterable, Tuple
from .typing import Array2D

from .geometry import Patch
from .util import ensure_ncomps



# Field types
# ----------------------------------------------------------------------


class FieldType:

    is_vector: bool
    is_displacement: bool

    @property
    def is_scalar(cls):
        return not cls.is_vector


class Scalar(FieldType):

    is_vector = False
    is_displacement = False


class Vector(FieldType):

    is_vector = True
    is_displacement = False


class Displacement(FieldType):

    is_vector = True
    is_displacement = True



# Abstract superclasses
# ----------------------------------------------------------------------


class FieldPatch(ABC):

    cells: bool
    name: str
    fieldtype: FieldType

    @property
    def is_scalar(self) -> bool:
        return self.fieldtype.is_scalar

    @property
    def is_vector(self) -> bool:
        return self.fieldtype.is_vector

    @property
    def is_displacement(self) -> bool:
        return self.fieldtype.is_displacement

    @property
    @abstractmethod
    def num_comps(self) -> int:
        """Number of components."""
        pass

    @abstractmethod
    def tesselate(self) -> Array2D:
        pass

    @abstractmethod
    def ensure_ncomps(self, ncomps: int, allow_scalar: bool = True):
        pass

    @abstractmethod
    def pick_component(self, index: int, name: str) -> 'FieldPatch':
        pass


class Field(ABC):

    name: str

    fieldtype: Optional[FieldType]

    # True if the field is defined on cells as opposed to nodes
    cells: bool

    # Number of components
    ncomps: int

    # True of vector fields can be decomposed to scalars
    decompose: bool

    @abstractmethod
    def patches(self, stepid: int, force: bool = False) -> Iterable[FieldPatch]:
        pass

    def decompositions(self) -> Iterable['Field']:
        if not self.decompose or self.ncomps == 1:
            return
        if self.ncomps > 3:
            log.warning(f"Attempted to decompose {self.name}, ignoring extra components")
        for index, suffix in zip(range(self.ncomps), 'xyz'):
            subname = f'{self.name}_{suffix}'
            yield ComponentField(subname, self, index)



# Component field
# ----------------------------------------------------------------------


class ComponentField(Field):

    ncomps = 1
    decompose = False
    fieldtype = Scalar()

    source: Field
    index: int

    def __init__(self, name: str, source: Field, index: int):
        self.name = name
        self.cells = source.cells
        self.source = source
        self.index = index

    def patches(self, stepid: int, force: bool = False) -> Iterable[FieldPatch]:
        for patch in self.source.patches(stepid, force=force):
            yield patch.pick_component(self.index, self.name)



# Combined field
# ----------------------------------------------------------------------


class CombinedField(Field):

    decompose = False

    sources: List[Field]

    def __init__(self, name: str, sources: List[Field]):
        self.name = name

        cells = set(source.cells for source in sources)
        if len(cells) > 1:
            sources = ', '.join(source.name for source in sources)
            raise TypeError(f"Attempted to combine incompatible fields: {sources}")
        self.cells = next(iter(cells))

        self.fieldtype = None
        self.ncomps = sum(source.ncomps for source in sources)
        self.sources = sources

    def patches(self, stepid: int, force: bool = False) -> Iterable[FieldPatch]:
        subpatch_iters = zip(*(source.patches(stepid, force=force) for source in self.sources))
        for subpatches in subpatch_iters:
            patches, data = [], []
            for fieldpatch in subpatches:
                if not isinstance(fieldpatch, SimpleFieldPatch):
                    raise TypeError(f"While forming combined field {self.name}, found nontrivial components")
                patches.append(fieldpatch.patch)
                data.append(fieldpatch.data)
            yield CombinedFieldPatch(self.name, patches, data)



# Simple field patch
# ----------------------------------------------------------------------


class SimpleFieldPatch(FieldPatch):

    patch: Patch
    data: Array2D

    def __init__(self, name: str, patch: Patch, data: Array2D, cells: bool = False, fieldtype: FieldType = None):
        self.name = name
        self.patch = patch
        self.data = data
        self.cells = cells
        if fieldtype is None:
            self.fieldtype = Vector() if self.num_comps > 1 else Scalar()
        else:
            self.fieldtype = fieldtype

    @property
    def num_comps(self) -> int:
        return self.data.shape[-1]

    def tesselate(self) -> Array2D:
        return self.patch.tesselate_field(self.data, cells=self.cells)

    def ensure_ncomps(self, ncomps: int, allow_scalar: bool = True):
        self.data = ensure_ncomps(self.data, ncomps, allow_scalar)

    def pick_component(self, index: int, name: str) -> FieldPatch:
        return SimpleFieldPatch(name, self.patch, self.data[:, index:index+1], cells=self.cells, fieldtype=Scalar())



# Combined field patch
# ----------------------------------------------------------------------


class CombinedFieldPatch(FieldPatch):

    patches: List[Patch]
    data: List[Array2D]

    def __init__(self, name: str, patches: List[Patch], data: List[Array2D], cells: bool = False, fieldtype: FieldType = None):
        self.name = name
        self.patches = patches
        self.data = data
        self.cells = cells
        if fieldtype is None:
            self.fieldtype = Vector() if self.num_comps > 1 else Scalar()
        else:
            self.fieldtype = fieldtype

    @property
    def num_comps(self) -> int:
        return sum(coeffs.shape[-1] for coeffs in self.data)

    def tesselate(self) -> Array2D:
        return np.hstack([
            patch.tesselate_field(coeffs, cells=self.cells)
            for patch, coeffs in zip(self.patches, self.data)
        ])

    def ensure_ncomps(self, ncomps: int, allow_scalar: bool = True):
        current_comps = self.num_comps
        if current_comps == 1 and allow_scalar:
            return
        if current_comps >= ncomps:
            return
        last = self.data[-1]
        self.data.append(np.zeros((last.shape[0], ncomps - current_comps), dtype=last.dtype))
        self.patches.append(self.patches[-1])

    def pick_component(self, index: int, name: str) -> FieldPatch:
        sourceid = 0
        while index >= self.data[sourceid].shape[1]:
            index -= self.data[sourceid].shape[1]
            sourceid += 1
        return SimpleFieldPatch(
            name, self.patches[sourceid], self.data[sourceid][:, index:index+1],
            cells=self.cells, fieldtype=Scalar()
        )
