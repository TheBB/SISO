from abc import ABC, abstractmethod
from functools import lru_cache
from io import BytesIO
from itertools import chain, count
from pathlib import Path

import h5py
import numpy as np
import treelog as log

from typing import Dict, Set, Optional, Any, Iterable, Tuple, List
from ..typing import Array2D, StepData, BoundingBox, PatchKey

from .reader import Reader
from .. import config, ConfigTarget
from ..coords import Local
from ..fields import Field, SimpleField, CombinedField, ComponentField, Displacement, Geometry, FieldPatches
from ..geometry import SplineTopology, LRTopology, UnstructuredTopology, Patch
from ..util import ensure_ncomps, bounding_box, cache
from ..writer import Writer



# PatchCatalogue
# ----------------------------------------------------------------------


class PatchCatalogue:

    ids: Dict[PatchKey, PatchKey]
    seqs: Dict[int, PatchKey]
    bboxes: Dict[BoundingBox, PatchKey]

    def __init__(self):
        self.bboxes = dict()
        self.seqs = dict()
        self.ids = dict()

    def setdefault(self, data: Array2D, oldkey: PatchKey) -> PatchKey:
        if oldkey in self.ids:
            return self.ids[oldkey]

        bbox = bounding_box(data)
        _, seq, *_ = oldkey

        try:
            newkey = self.bboxes[bbox]
            log.debug(f"Patch {oldkey} identified with {newkey} by bounding box")
        except KeyError:
            if config.strict_id:
                newkey = oldkey
            else:
                try:
                    newkey = self.seqs[seq]
                    log.debug(f"Patch {oldkey} identified with {newkey} by sequence number")
                except KeyError:
                    newkey = oldkey

        self.ids[oldkey] = newkey
        self.bboxes[bbox] = newkey
        self.seqs[seq] = newkey
        return newkey



# Basis
# ----------------------------------------------------------------------


class Basis(ABC):

    name: str
    reader: 'IFEMReader'

    npatches: int

    def __init__(self, name: str, reader: 'IFEMReader'):
        self.name = name
        self.reader = reader

    @abstractmethod
    def group_path(self, stepid: int) -> str:
        pass

    @abstractmethod
    def update_at(self, stepid: int) -> bool:
        pass

    @property
    @abstractmethod
    def num_updates(self) -> int:
        pass

    @cache(1)
    def patch_at(self, stepid: int, patchid: int) -> Tuple[Patch, Array2D]:
        while not self.update_at(stepid):
            stepid -= 1

        subpath = self.group_path(stepid)
        patchdata = self.reader.h5[f'{subpath}/{patchid+1}'][:]
        initial = patchdata[:20].tobytes()
        g2bytes = BytesIO(memoryview(patchdata))
        if initial.startswith(b'# LAGRANGIAN'):
            topo, nodes = UnstructuredTopology.from_lagrangian(g2bytes)
        elif initial.startswith(b'# LRSPLINE'):
            topo, nodes = next(LRTopology.from_string(g2bytes.read()))
        else:
            topo, nodes = next(SplineTopology.from_string(g2bytes.read()))

        oldkey = (self.name, patchid)
        newkey = self.reader.patch_catalogue.setdefault(nodes, oldkey)
        return Patch(newkey, topo), nodes



class StandardBasis(Basis):

    name: str
    reader: 'IFEMReader'

    update_steps: Set[int]
    npatches: int

    def __init__(self, name: str, reader: 'IFEMReader'):
        super().__init__(name, reader)

        # Find out at which steps this basis updates, and how many patches it has
        self.update_steps = set()
        self.npatches = 0
        for i, _ in reader.steps():
            subpath = self.group_path(i)
            if subpath not in reader.h5:
                continue
            self.update_steps.add(i)
            self.npatches = max(self.npatches, len(reader.h5[subpath]))

    def group_path(self, stepid: int) -> str:
        return f'{self.reader.stepgroup[stepid]}/{self.name}/basis'

    def update_at(self, stepid: int) -> bool:
        return stepid in self.update_steps

    @property
    def num_updates(self) -> int:
        return len(self.update_steps)


class EigenBasis(Basis):

    def __init__(self, name: str, reader: 'IFEMReader'):
        super().__init__(name, reader)
        self.npatches = len(reader.h5[self.group_path(0)])

    def group_path(self, stepid: int) -> str:
        return f'0/{self.name}/basis'

    def update_at(self, stepid: int) -> bool:
        return stepid == 0

    @property
    def num_updates(self) -> int:
        return 1



# Fields
# ----------------------------------------------------------------------


class IFEMGeometryField(SimpleField):

    basis: Basis

    cells = False

    def __init__(self, basis: Basis):
        self.name = basis.name
        self.basis = basis
        self.fieldtype = Geometry(Local(basis.name).substitute())

    def patches(self, stepid: int, force: bool = False, **_) -> FieldPatches:
        if not force and not self.basis.update_at(stepid):
            return
        for patchid in range(self.basis.npatches):
            patch, coeffs = self.basis.patch_at(stepid, patchid)
            yield patch, coeffs


class IFEMField(SimpleField):

    basis: Basis
    decompose = True

    def __init__(self, name: str, basis: Basis, reader: 'IFEMReader', cells: bool = False):
        self.name = name
        self.basis = basis
        self.cells = cells
        self.fieldtype = None
        self.reader = reader

        # Calculate number of components
        stepid = next(i for i in count() if self.update_at(i))
        patch, _ = self.basis.patch_at(stepid, 0)
        denominator = patch.topology.num_cells if cells else patch.topology.num_nodes
        ncoeffs = len(self.reader.h5[self.coeff_path(stepid, 0)])
        if ncoeffs % denominator != 0:
            raise ValueError(
                f"Inconsistent dimension in field '{self.name}' ({ncoeffs}/{denominator}); "
                "unable to discover number of components"
            )
        self.ncomps = ncoeffs // denominator

    @property
    def basisname(self) -> str:
        return self.basis.name

    def patches(self, stepid: int, force: bool = False, **_) -> FieldPatches:
        if not force and not self.update_at(stepid):
            return
        for patchid in range(self.basis.npatches):
            patch, _ = self.basis.patch_at(stepid, patchid)
            coeffs = self.coeffs(stepid, patchid)
            yield patch, coeffs

    def update_at(self, stepid: int) -> bool:
        return self.group_path(stepid) in self.reader.h5

    def group_path(self, stepid: int) -> str:
        celltype = 'knotspan' if self.cells else 'fields'
        return f'{self.reader.stepgroup[stepid]}/{self.basisname}/{celltype}/{self.name}'

    def coeff_path(self, stepid: int, patchid: int) -> str:
        return f'{self.group_path(stepid)}/{patchid+1}'

    def coeffs(self, stepid: int, patchid: int) -> Array2D:
        coeffs = self.reader.h5[self.coeff_path(stepid, patchid)][:]
        return coeffs.reshape((-1, self.ncomps))


class EigenField(IFEMField):

    def __init__(self, name: str, basis: Basis, reader: 'IFEMReader'):
        super().__init__(name, basis, reader, cells=False)
        self.fieldtype = Displacement()
        self.decompose = False

    def group_path(self, stepid: int) -> str:
        return f'0/{self.basis.name}/Eigenmode/{stepid+1}'

    def coeff_path(self, stepid: int, patchid: int) -> str:
        return f'{self.group_path(stepid)}/{patchid+1}'

    def coeffs(self, stepid: int, patchid: int) -> Array2D:
        coeffs = super().coeffs(stepid, patchid)
        coeffs = ensure_ncomps(coeffs, 3, False)
        if self.ncomps == 1:
            coeffs[:, -1] = coeffs[:, 0].copy()
            coeffs[:, 0] = 0
        return coeffs



# Reader classes
# ----------------------------------------------------------------------


class IFEMReader(Reader):

    reader_name = "IFEM"

    filename: Path
    h5: h5py.File

    bases: Dict[str, Basis]
    _fields: Dict[str, Field]
    _field_basis: Dict[str, str]
    stepgroup: List[int]

    patch_catalogue: PatchCatalogue

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        """Check if it's a valid HDF5 file and that it contains a group called '0'."""
        try:
            with h5py.File(filename, 'r') as f:
                list(map(int, f))   # All groups must have integer names
            return True
        except:
            return False

    def __init__(self, filename: Path):
        self.filename = filename
        self.bases = dict()
        self._fields = dict()
        self._field_basis = dict()
        self.patch_catalogue = PatchCatalogue()

    def validate(self):
        super().validate()
        config.ensure_limited(
            ConfigTarget.Reader, 'only_bases', 'strict_id',
            reason="not supported by IFEM"
        )

    def __enter__(self):
        self.h5 = h5py.File(str(self.filename), 'r').__enter__()
        self.stepgroup = sorted(list(map(int, self.h5)))

        # Populate self.bases
        self.init_bases()

        # Populate self._fields
        # This is a complicated process broken down into steps
        self.init_fields()
        self.split_fields()
        self.combine_fields()

        return self

    def __exit__(self, *args):
        self.h5.__exit__(*args)

    @property
    def nsteps(self) -> int:
        """Return number of steps in the data set."""
        return len(self.h5)

    def stepdata(self, stepid: int) -> StepData:
        """Return the data associated with a step (time, eigenvalue or
        frequency).
        """
        try:
            time = self.h5[f'{self.stepgroup[stepid]}/timeinfo/level'][0]
        except KeyError:
            time = float(stepid)
        return {'time': time}

    def steps(self) -> Iterable[Tuple[int, StepData]]:
        """Yield a sequence of step IDs."""
        for stepid in range(self.nsteps):
            yield stepid, self.stepdata(stepid)

    def field_basis(self, fieldname: str) -> Basis:
        return self.bases[self._field_basis[fieldname]]

    def init_bases(self, basisnames: Optional[Set[str]] = None, constructor: type = StandardBasis):
        """Populate the contents of self.bases.

        To allow easier subclassing, the two keyword arguments allows
        a subclass to override which bases and which basis
        class to use.
        """
        if basisnames is None:
            basisnames = set(chain.from_iterable(self.h5.values()))

        # Construct Basis objects for each discovered basis name
        for basisname in basisnames:
            self.bases[basisname] = constructor(basisname, self)

        # Delete timeinfo, if present
        if 'timeinfo' in self.bases:
            del self.bases['timeinfo']

        # Delete bases that don't have any patch data
        to_del = [name for name, basis in self.bases.items() if basis.num_updates == 0]
        for basisname in to_del:
            log.debug(f"Removing basis {basisname}: no updates")
            del self.bases[basisname]

        # Delete the bases we don't need
        if config.only_bases:
            keep = {b.lower() for b in config.only_bases} | {config.coords.name.lower()}
            self.bases = {name: basis for name, basis in self.bases.items() if name.lower() in keep}

        # Debug output
        for basis in self.bases.values():
            log.debug(f"Basis {basis.name} updates at {basis.num_updates} step(s) with {basis.npatches} patch(es)")

    def init_fields(self):
        """Discover fields and populate the contents of self.fields."""
        for stepid, _ in self.steps():
            stepgrp = self.h5[str(self.stepgroup[stepid])]
            for basisname, basisgrp in stepgrp.items():
                if basisname not in self.bases:
                    continue

                for fieldname in basisgrp.get('fields', ()):
                    if fieldname not in self._fields and basisname in self.bases:
                        if config.field_filter is not None and fieldname.lower() not in config.field_filter:
                            continue
                        self._fields[fieldname] = IFEMField(fieldname, self.bases[basisname], self)
                        self._field_basis[fieldname] = basisname
                for fieldname in basisgrp.get('knotspan', ()):
                    if fieldname not in self._fields and basisname in self.bases:
                        if config.field_filter is not None and fieldname.lower() not in config.field_filter:
                            continue
                        self._fields[fieldname] = IFEMField(fieldname, self.bases[basisname], self, cells=True)
                        self._field_basis[fieldname] = basisname

    def split_fields(self):
        """Split fields which are stored inline to separate scalar fields,
        e.g. u_y&&T -> u_y, T .
        """
        to_add, to_del = [], []
        for fname in self._fields:
            if '&&' not in fname:
                continue
            splitnames = [s.strip() for s in fname.split('&&')]

            # Check if the first name has a prefix, if so apply it to all the names
            if ' ' in splitnames[0]:
                prefix, splitnames[0] = splitnames[0].split(' ', 1)
                splitnames = ['{} {}'.format(prefix, name) for name in splitnames]

            if any(splitname in self._fields for splitname in splitnames):
                log.warning(f"Unable to split '{fname}', some fields already exist".format(fname))
                continue

            for i, splitname in enumerate(splitnames):
                to_add.append(ComponentField(splitname, self._fields[fname], i))
                self._field_basis[splitname] = self._field_basis[fname]
            to_del.append(fname)

        for fname in to_del:
            del self._fields[fname]
        for field in to_add:
            self._fields[field.name] = field

    def combine_fields(self):
        """Combine fields which have the same suffix to a unified vectorized field, e.g.
        u_x, u_y, u_z -> u.
        """
        candidates = dict()
        for fname in self._fields:
            if len(fname) > 2 and fname[-2] == '_' and fname[-1] in 'xyz' and fname[:-2] not in self._fields:
                candidates.setdefault(fname[:-2], []).append(fname)

        for fname, sourcenames in candidates.items():
            if not (1 < len(sourcenames) < 4):
                continue

            sourcenames = sorted(sourcenames, key=lambda s: s[-1])
            sources = [self._fields[s] for s in sourcenames]
            sourcenames = ', '.join(sourcenames)
            self._fields[fname] = CombinedField(fname, sources)
            log.info(f"Creating combined field {sourcenames} -> {fname}")

    def fields(self) -> Iterable[Field]:
        for basis in self.bases.values():
            yield IFEMGeometryField(basis)
        for field in self._fields.values():
            if field.ncomps > 0:
                yield field


class IFEMEigenReader(IFEMReader):

    reader_name = "IFEM-eigenmodes"

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        try:
            with h5py.File(filename, 'r') as f:
                assert '0' in f
                basisname = next(iter(f['0']))
                assert 'Eigenmode' in f['0'][basisname]
            return True
        except:
            return False

    @property
    def basis_group(self):
        return next(iter(self.h5['0'].values()))

    @property
    def nsteps(self):
        return len(self.basis_group['Eigenmode'])

    def stepdata(self, stepid: int) -> Dict[str, Any]:
        grp = self.basis_group[f'Eigenmode/{stepid+1}']
        if 'Value' in grp:
            return {'value': grp['Value'][0]}
        return {'frequency': grp['Frequency'][0]}

    def init_bases(self):
        basisname = next(iter(self.h5['0']))
        super().init_bases(basisnames={basisname}, constructor=EigenBasis)

    def init_fields(self):
        basis = next(iter(self.bases.values()))
        self._fields['Mode Shape'] = EigenField('Mode Shape', basis, self)
