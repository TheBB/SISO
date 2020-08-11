from abc import ABC, abstractmethod
from functools import lru_cache
from itertools import chain
from pathlib import Path

import h5py
import numpy as np
import treelog as log

from typing import Dict, Set, Optional, List, Any, Iterable, Tuple
from ..typing import Array2D

from .reader import Reader
from .. import config
from ..util import ensure_ncomps
from ..fields import SimpleFieldPatch, CombinedFieldPatch, FieldType, Scalar, Displacement
from ..geometry import Patch, SplinePatch, LRPatch, UnstructuredPatch
from ..writer import Writer



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

    @lru_cache(24)
    def patch_at(self, stepid: int, patchid: int) -> Patch:
        while not self.update_at(stepid):
            stepid -= 1
        subpath = self.group_path(stepid)
        g2bytes = self.reader.h5[f'{subpath}/{patchid+1}'][:].tobytes()
        patchkey = (self.name, patchid)
        if g2bytes.startswith(b'# LAGRANGIAN'):
            return UnstructuredPatch.from_lagrangian(patchkey, g2bytes)
        elif g2bytes.startswith(b'# LRSPLINE'):
            return LRPatch(patchkey, g2bytes)
        else:
            return SplinePatch(patchkey, g2bytes)


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
        for i in reader.steps():
            subpath = self.group_path(i)
            if subpath not in reader.h5:
                continue
            self.update_steps.add(i)
            self.npatches = max(self.npatches, len(reader.h5[subpath]))

    def group_path(self, stepid: int) -> str:
        return f'{stepid}/{self.name}/basis'

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



# Field superclasses
# ----------------------------------------------------------------------


class Field(ABC):

    name: str

    # True if the field is defined on cells as opposed to nodes
    cells: bool

    # Number of components in the source data
    ncomps: int

    def __init__(self, name: str, cells: bool):
        self.name = name
        self.cells = cells

    @property
    @abstractmethod
    def basisname(self) -> str:
        pass

    @abstractmethod
    def update_at(self, stepid: int) -> bool:
        pass

    @abstractmethod
    def update(self, w: Writer, stepid: int):
        pass


class SimpleField(Field):

    basis: Basis

    # True if vector-valued fields can be decomposed into scalars
    decompose: bool = True

    fieldtype: Optional[FieldType]

    @property
    def basisname(self) -> str:
        return self.basis.name

    def update(self, w: Writer, stepid: int):
        for patchid in range(self.basis.npatches):
            patch = self.basis.patch_at(stepid, patchid)
            coeffs = self.coeffs(stepid, patchid)
            w.update_field(SimpleFieldPatch(self.name, patch, coeffs, cells=self.cells, fieldtype=self.fieldtype))

            if self.decompose and self.ncomps > 1:
                for i, subscript in zip(range(self.ncomps), 'xyz'):
                    w.update_field(SimpleFieldPatch(
                        f'{self.name}_{subscript}', patch, coeffs[...,i:i+1], cells=self.cells
                    ))

    @abstractmethod
    def coeffs(self, stepid: int, patchid: int) -> Array2D:
        pass



# Concrete field classes
# ----------------------------------------------------------------------


class StandardField(SimpleField):

    update_steps: Set[int]

    def __init__(self, name: str, basis: Basis, reader: 'IFEMReader', cells: bool = False):
        self.name = name
        self.basis = basis
        self.cells = cells
        self.fieldtype = None
        self.reader = reader

        # Find out at which steps this field updates
        subpath = self.group_path
        self.update_steps = {i for i in reader.steps() if self.group_path(i) in reader.h5}

        # Calculate number of components
        stepid = next(iter(self.update_steps))
        patch = self.basis.patch_at(stepid, 0)
        denominator = patch.num_cells if cells else patch.num_nodes
        ncoeffs = len(self.reader.h5[self.coeff_path(stepid, 0)])
        if ncoeffs % denominator != 0:
            raise ValueError(
                f"Inconsistent dimension in field '{self.name}' ({ncoeffs}/{denominator}); "
                "unable to discover number of components"
            )
        self.ncomps = ncoeffs // denominator

    def update_at(self, stepid: int) -> bool:
        return stepid in self.update_steps

    def group_path(self, stepid: int) -> str:
        celltype = 'knotspan' if self.cells else 'fields'
        return f'{stepid}/{self.basisname}/{celltype}/{self.name}'

    def coeff_path(self, stepid: int, patchid: int) -> str:
        return f'{self.group_path(stepid)}/{patchid+1}'

    def coeffs(self, stepid: int, patchid: int) -> Array2D:
        coeffs = self.reader.h5[self.coeff_path(stepid, patchid)][:]
        return coeffs.reshape((-1, self.ncomps))


class CombinedField(Field):

    sources: List[SimpleField]

    def __init__(self, name: str, sources: List[SimpleField]):
        cells = set(s.cells for s in sources)
        assert len(cells) == 1

        self.name = name
        self.cells = next(iter(cells))
        self.ncomps = sum(src.ncomps for src in sources)
        self.sources = sources

    @property
    def basisname(self) -> str:
        return ','.join(source.basisname for source in self.sources)

    def update_at(self, stepid: int) -> bool:
        return any(src.update_at(stepid) for src in self.sources)

    def update(self, w, stepid):
        # Assume all bases have the same number of patches,
        # so just grab the indices from the first one

        for patchid in range(self.sources[0].basis.npatches):
            patches, results = [], []
            for src in self.sources:
                patch = src.basis.patch_at(stepid, patchid)
                patches.append(patch)
                coeffs = src.coeffs(stepid, patchid)
                results.append(coeffs)

            w.update_field(CombinedFieldPatch(self.name, patches, results, cells=self.cells))


class SplitField(SimpleField):

    source: SimpleField
    index: int

    def __init__(self, name, source, index):
        self.name = name
        self.source = source
        self.index = index
        self.ncomps = 1

        self.basis = source.basis
        self.cells = source.cells
        self.decompose = False
        self.fieldtype = Scalar()

    def update_at(self, stepid: int) -> bool:
        return self.source.update_at(stepid)

    def coeffs(self, stepid: int, patchid: int) -> Array2D:
        coeffs = self.source.coeffs(stepid, patchid)
        return coeffs[:, self.index:self.index+1]



# Geometry manager
# ----------------------------------------------------------------------


class GeometryManager:

    basis: Basis
    written: bool

    def __init__(self, basis: Basis):
        self.basis = basis
        self.written = False
        log.info(f"Using {basis.name} for geometry")

    def update(self, w: Writer, stepid: int):
        if not self.basis.update_at(stepid) and self.written:
            w.finalize_geometry()
            return

        log.info("Updating geometry")

        # FIXME: Here, all patches are updated whenever any of them are updated.
        # Maybe overkill.
        for patchid in range(self.basis.npatches):
            patch = self.basis.patch_at(stepid, patchid)
            w.update_geometry(patch)

        w.finalize_geometry()
        self.written = True



# Reader classes
# ----------------------------------------------------------------------


class IFEMReader(Reader):

    reader_name = "IFEM"

    filename: Path
    h5: h5py.File

    bases: Dict[str, Basis]
    fields: Dict[str, Field]

    @classmethod
    def applicable(cls, filename: Path) -> bool:
        """Check if it's a valid HDF5 file and that it contains a group called '0'."""
        try:
            with h5py.File(filename, 'r') as f:
                assert '0' in f
            return True
        except:
            return False

    def __init__(self, filename: Path):
        self.filename = filename
        self.bases = dict()
        self.fields = dict()

    def __enter__(self):
        self.h5 = h5py.File(str(self.filename), 'r').__enter__()

        # Populate self.bases
        self.init_bases()

        # Populate self.fields
        # This is a complicated process broken down into steps
        self.init_fields()
        self.log_fields()
        self.split_fields()
        self.combine_fields()
        self.sort_fields()

        # Create geometry manager
        geometry_basis = config.geometry_basis or next(iter(self.bases))
        self.geometry = GeometryManager(self.bases[geometry_basis])

        return self

    def __exit__(self, *args):
        self.h5.__exit__(*args)

    @property
    def nsteps(self) -> int:
        """Return number of steps in the data set."""
        return len(self.h5)

    def steps(self) -> Iterable[int]:
        """Yield a sequence of step IDs."""
        yield from range(self.nsteps)

    def stepdata(self, stepid: int) -> Dict[str, Any]:
        """Return the data associated with a step (time, eigenvalue or
        frequency).
        """
        try:
            time = self.h5[f'{stepid}/timeinfo/level'][0]
        except KeyError:
            time = float(stepid)
        return {'time': time}

    def outputsteps(self) -> Iterable[Tuple[int, Dict[str, Any]]]:
        """Yield an iterator of timesteps to be sent to the writer, together
        with step data.  This obeys the setting of
        config.only_final_timestep."""
        if config.only_final_timestep:
            for stepid in self.steps():
                pass
            yield stepid, self.stepdata(stepid)
        else:
            for stepid in self.steps():
                yield stepid, self.stepdata(stepid)

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
            keep = set(config.only_bases)
            if config.geometry_basis:
                keep.add(config.geometry_basis)
            self.bases = {name: basis for name, basis in self.bases.items() if name in keep}

        # Debug output
        for basis in self.bases.values():
            log.debug(f"Basis {basis.name} updates at {basis.num_updates} step(s) with {basis.npatches} patch(es)")

    def init_fields(self):
        """Discover fields and populate the contents of self.fields."""
        for stepid in self.steps():
            stepgrp = self.h5[str(stepid)]
            for basisname, basisgrp in stepgrp.items():
                if basisname not in self.bases:
                    continue

                for fieldname in basisgrp.get('fields', ()):
                    if fieldname not in self.fields and basisname in self.bases:
                        self.fields[fieldname] = StandardField(fieldname, self.bases[basisname], self)
                for fieldname in basisgrp.get('knotspan', ()):
                    if fieldname not in self.fields and basisname in self.bases:
                        self.fields[fieldname] = StandardField(fieldname, self.bases[basisname], self, cells=True)

    def log_fields(self):
        """Print brief field information to log.debug."""
        for field in self.fields.values():
            log.debug(f"Field '{field.name}' lives on {field.basisname} with {field.ncomps} component(s)")

    def split_fields(self):
        """Split fields which are stored inline to separate scalar fields,
        e.g. u_y&&T -> u_y, T .
        """
        to_add, to_del = [], []
        for fname in self.fields:
            if '&&' not in fname:
                continue
            splitnames = [s.strip() for s in fname.split('&&')]

            # Check if the first name has a prefix, if so apply it to all the names
            if ' ' in splitnames[0]:
                prefix, splitnames[0] = splitnames[0].split(' ', 1)
                splitnames = ['{} {}'.format(prefix, name) for name in splitnames]

            if any(splitname in self.fields for splitname in splitnames):
                log.warning(f"Unable to split '{fname}', some fields already exist".format(fname))
                continue

            for i, splitname in enumerate(splitnames):
                to_add.append(SplitField(splitname, self.fields[fname], i))
            to_del.append(fname)

        for fname in to_del:
            del self.fields[fname]
        for field in to_add:
            self.fields[field.name] = field

    def combine_fields(self):
        """Combine fields which have the same suffix to a unified vectorized field, e.g.
        u_x, u_y, u_z -> u.
        """
        candidates = dict()
        for fname in self.fields:
            if len(fname) > 2 and fname[-2] == '_' and fname[-1] in 'xyz' and fname[:-2] not in self.fields:
                candidates.setdefault(fname[:-2], []).append(fname)

        for fname, sourcenames in candidates.items():
            if not (1 < len(sourcenames) < 4):
                continue

            sourcesnames = sorted(sourcenames, key=lambda s: s[-1])
            sources = [self.fields[s] for s in sourcenames]
            sourcenames = ', '.join(sourcenames)
            try:
                self.fields[fname] = CombinedField(fname, sources)
                log.info(f"Creating combined field {sourcenames} -> {fname}")
            except AssertionError:
                log.warning("Unable to combine fields {sourcenames} -> {fname}")

    def sort_fields(self):
        """Sort fields by name."""
        fields = sorted(self.fields.values(), key=lambda f: f.name)
        fields = sorted(fields, key=lambda f: f.cells)
        self.fields = {f.name: f for f in fields}

    def write(self, w: Writer):
        for stepid, time in log.iter.plain('Step', self.outputsteps()):
            w.add_step(**time)
            self.geometry.update(w, stepid)

            for field in self.fields.values():
                if field.update_at(stepid) or config.only_final_timestep:
                    log.info('Updating {} ({})'.format(field.name, field.basisname))
                    field.update(w, stepid)

            w.finalize_step()


class EigenField(StandardField):

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
        self.fields['Mode Shape'] = EigenField('Mode Shape', basis, self)
