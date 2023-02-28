from __future__ import annotations

from functools import lru_cache
from itertools import chain, count, repeat
import logging
from pathlib import Path

import h5py
import numpy as np

from ..api import (
    ReaderSettings,
    RecombineFieldSpec,
    Source,
    SourceProperties,
    SplitFieldSpec,
    Topology,
)
from ..field import Field, FieldType, FieldData
from ..timestep import TimeStep
from ..topology import SplineTopology, LrTopology, UnstructuredTopology
from ..zone import Zone, Shape
from .. import util

from typing_extensions import Self
from typing import (
    ClassVar,
    Dict,
    Iterator,
    Optional,
    Protocol,
    List,
    Set,
    Tuple,
)


class Locator(Protocol):
    def patch_path(self, name: str, step: int, patch: int) -> str:
        ...

    def coeff_path(self, basis_name: str, field_name: str, step: int, patch: int, cellwise: bool) -> str:
        ...


class StandardLocator:
    def patch_path(self, name: str, step: int, patch: int) -> str:
        return f'{step}/{name}/basis//{patch+1}'

    def coeff_path(self, basis_name: str, field_name: str, step: int, patch: int, cellwise: bool) -> str:
        subdir = 'knotspan' if cellwise else 'fields'
        return f'{int(step)}/{basis_name}/{subdir}/{field_name}/{patch+1}'


class EigenLocator:
    def patch_path(self, name: str, step: int, patch: int) -> str:
        return f'0/{name}/basis/{patch+1}'

    def coeff_path(self, basis_name: str, field_name: str, step: int, patch: int, cellwise: bool) -> str:
        return f'0/{basis_name}/Eigenmode/{step+1}/{patch+1}'


def is_legal_group_name(name: str) -> bool:
    try:
        int(name)
        return True
    except ValueError:
        return name.lower() in ('anasol', 'log')


class IfemBasis:
    name: str
    num_patches: int

    locator: Locator

    def __init__(self, name: str, locator: Locator, source: Ifem):
        self.name = name
        self.locator = locator
        self.num_patches = 0

        i = 0
        for i in count():
            if self.patch_path(0, i) not in source.h5:
                break
        self.num_patches = i

    def __repr__(self) -> str:
        return f'Basis({self.name}, num_patches={self.num_patches})'

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if isinstance(other, IfemBasis):
            return self.name == other.name
        return False

    def patch_path(self, step: int, patch: int) -> str:
        return self.locator.patch_path(self.name, step, patch)

    @lru_cache(maxsize=128)
    def patch_at(self, step: int, patch: int, source: Ifem) -> Tuple[Zone, Topology, FieldData]:
        while self.patch_path(step, patch) not in source.h5:
            step -= 1

        topology: Topology
        patchdata = source.h5[self.patch_path(step, patch)][:]
        initial = patchdata[:20].tobytes()
        raw_data = memoryview(patchdata).tobytes()
        if initial.startswith(b'# LAGRANGIAN'):
            corners, topology, cps = UnstructuredTopology.from_ifem(raw_data)
        elif initial.startswith(b'# LRSPLINE'):
            corners, topology, cps = next(LrTopology.from_bytes(raw_data))
        else:
            corners, topology, cps = next(SplineTopology.from_bytes(raw_data))
        shape = [Shape.Line, Shape.Quatrilateral, Shape.Hexahedron][topology.pardim - 1]

        zone = Zone(shape=shape, coords=corners, local_key=f'{self.name}/{step}/{patch}')
        return zone, topology, cps

    @lru_cache(maxsize=1)
    def ncomps(self, source: Ifem) -> int:
        _, _, cps = self.patch_at(0, 0, source)
        return cps.ncomps


class IfemField:
    name: str
    cellwise: bool
    basis: IfemBasis

    source_fields: List[str]
    components: List[Optional[List[int]]]

    def __init__(self, name: str, cellwise: bool, basis: IfemBasis):
        self.name = name
        self.cellwise = cellwise
        self.basis = basis

    def __repr__(self) -> str:
        return f"Field({self.name}, {'cellwise' if self.cellwise else 'nodal'}, {self.basis.name})"

    def splits(self) -> Iterator[SplitFieldSpec]:
        if '&&' not in self.name:
            return

        component_names = [comp.strip() for comp in self.name.split('&&')]
        if ' ' in component_names[0]:
            prefix, component_names[0] = component_names[0].split(' ', maxsplit=1)
            component_names = [f'{prefix} {comp}' for comp in component_names]

        for i, comp in enumerate(component_names):
            yield SplitFieldSpec(
                source_name=self.name,
                new_name=comp,
                components=[i],
                destroy=True,
            )

    def patch_path(self, step: int, patch: int) -> str:
        return self.basis.locator.coeff_path(
            self.basis.name,
            self.name,
            step,
            patch,
            self.cellwise,
        )

    @lru_cache(maxsize=1)
    def ncomps(self, source: Ifem) -> int:
        _, topology, _ = self.basis.patch_at(0, 0, source)
        my_cps = self.raw_cps_at(0, 0, source)
        divisor = topology.num_cells if self.cellwise else topology.num_nodes
        ncomps, remainder = divmod(len(my_cps), divisor)
        assert remainder == 0
        return ncomps

    @lru_cache(maxsize=8)
    def raw_cps_at(self, step: int, patch: int, source: Ifem) -> np.ndarray:
        return source.h5[self.patch_path(step, patch)][:]

    def cps_at(self, step: int, patch: int, source: Ifem) -> FieldData:
        ncomps = self.ncomps(source)
        cps = self.raw_cps_at(step, patch, source)
        return FieldData(data=cps.reshape(-1, ncomps))


class Ifem(Source):
    filename: Path
    h5: h5py.File

    geometry: IfemBasis

    _bases: Dict[str, IfemBasis]
    _fields: Dict[str, IfemField]

    locator: ClassVar[Locator] = StandardLocator()
    default_field_type: ClassVar[FieldType] = FieldType.Generic

    @staticmethod
    def applicable(path: Path) -> bool:
        try:
            with h5py.File(path, 'r') as f:
                assert all(is_legal_group_name(name) for name in f)
            return True
        except (AssertionError, OSError):
            return False

    def __init__(self, filename: Path):
        self.filename = filename
        self._fields = {}

    def __enter__(self) -> Self:
        self.h5 = h5py.File(self.filename, 'r').__enter__()

        self.discover_bases()
        for basis in self._bases.values():
            logging.debug(
                f"Basis {basis.name} with "
                f"{util.pluralize(basis.num_patches, 'patch', 'patches')}"
            )

        self.discover_fields()
        return self

    def __exit__(self, *args) -> None:
        self.h5.__exit__(*args)

    @property
    def properties(self) -> SourceProperties:
        splits, recombineations = self.propose_recombinations()
        return SourceProperties(
            instantaneous=False,
            split_fields=splits,
            recombine_fields=recombineations,
        )

    def configure(self, settings: ReaderSettings) -> None:
        return

    @property
    def nsteps(self) -> int:
        return len(self.h5)

    def timestep_groups(self) -> Iterator[h5py.Group]:
        for index in range(self.nsteps):
            yield self.h5[str(index)]

    def make_basis(self, name: str) -> IfemBasis:
        return IfemBasis(name, self.locator, self)

    def discover_bases(self) -> None:
        basis_names = set(chain.from_iterable(self.h5.values())) - {'timeinfo'}
        bases = (self.make_basis(name) for name in basis_names)
        self._bases = {
            basis.name: basis
            for basis in bases
            if basis.num_patches > 0
        }

    def discover_fields(self) -> None:
        for step_grp in self.timestep_groups():
            for basis_name, basis_grp in step_grp.items():
                if basis_name not in self._bases:
                    continue

                fields: Iterator[Tuple[str, bool]] = chain(
                    zip(basis_grp.get('fields', ()), repeat(False)),
                    zip(basis_grp.get('knotspan', ()), repeat(True)),
                )
                for field_name, cellwise in fields:
                    self._fields[field_name] = IfemField(
                        name=field_name,
                        cellwise=cellwise,
                        basis=self._bases[basis_name],
                    )

    @lru_cache(maxsize=1)
    def propose_recombinations(self) -> Tuple[List[SplitFieldSpec], List[RecombineFieldSpec]]:
        splits = list(chain.from_iterable(field.splits() for field in self._fields.values()))

        candidates: Dict[str, List[str]] = {}
        field_names = chain(self._fields, (split.new_name for split in splits))
        for field_name in field_names:
            if len(field_name) <= 2 or field_name[-2] != '_':
                continue
            prefix, suffix = field_name[:-2], field_name[-1]
            if suffix not in 'xyz':
                continue
            candidates.setdefault(prefix, []).append(field_name)

        recombinations = [
            RecombineFieldSpec(source_names, new_name)
            for new_name, source_names in candidates.items()
            if new_name not in self._fields and len(source_names) > 1
        ]

        return splits, recombinations

    def use_geometry(self, geometry: Field) -> None:
        self.geometry = self._bases[geometry.name]

    def timesteps(self) -> Iterator[TimeStep]:
        for i, group in enumerate(self.timestep_groups()):
            if 'timeinfo/level' in group:
                time = group['timeinfo/level']
            else:
                time = float(i)
            yield TimeStep(index=i, time=time)

    def zones(self) -> Iterator[Zone]:
        for patch in range(self.geometry.num_patches):
            zone, _, _ = self.geometry.patch_at(0, patch, self)
            yield zone

    def fields(self) -> Iterator[Field]:
        for basis in self._bases.values():
            yield Field(
                name=basis.name,
                type=FieldType.Geometry,
                ncomps=basis.ncomps(self),
                cellwise=False,
            )

        for field in self._fields.values():
            yield Field(
                name=field.name,
                type=self.default_field_type,
                ncomps=field.ncomps(self),
                cellwise=field.cellwise,
            )

    def topology(self, timestep: TimeStep, field: Field, zone: Zone) -> Topology:
        if field.type == FieldType.Geometry:
            basis = self._bases[field.name]
        else:
            basis = self._fields[field.name].basis
        patch = int(zone.local_key.split('/')[-1])
        _, topology, _ = basis.patch_at(timestep.index, patch, self)
        return topology

    def field_data(self, timestep: TimeStep, field: Field, zone: Zone) -> FieldData:
        patch = int(zone.local_key.split('/')[-1])
        if field.type == FieldType.Geometry:
            basis = self._bases[field.name]
            _, _, coeffs = basis.patch_at(timestep.index, patch, self)
            return coeffs
        ifield = self._fields[field.name]
        coeffs = ifield.cps_at(timestep.index, patch, self)
        if field.type == FieldType.Eigenmode:
            coeffs = coeffs.ensure_ncomps(3, allow_scalar=False, pad_right=False)
        return coeffs


class IfemModes(Ifem):

    locator = EigenLocator()
    default_field_type = FieldType.Eigenmode

    @staticmethod
    def applicable(path: Path) -> bool:
        try:
            with h5py.File(path, 'r') as f:
                assert '0' in f
                basis_name = next(iter(f['0']))
                assert 'Eigenmode' in f[f'0/{basis_name}']
            return True
        except (AssertionError, OSError):
            return False

    def discover_bases(self) -> None:
        group = self.h5['0']
        basis_name = util.only(group)
        self._bases = {
            basis_name: self.make_basis(basis_name)
        }

    def discover_fields(self) -> None:
        basis = util.only(self._bases.values())
        self._fields = {
            'Mode Shape': IfemField('Mode Shape', cellwise=False, basis=basis)
        }

    @property
    def nsteps(self) -> int:
        basis = util.only(self._bases.values())
        return len(self.h5[f'0/{basis.name}/Eigenmode'])

    def timestep_groups(self) -> Iterator[h5py.Group]:
        basis = util.only(self._bases.values())
        for index in range(self.nsteps):
            yield self.h5[f'0/{basis.name}/Eigenmode/{index+1}']

    def timesteps(self) -> Iterator[TimeStep]:
        for i, group in enumerate(self.timestep_groups()):
            if 'Value' in group:
                time = group['Value'][0]
            else:
                time = group['Frequency'][0]
            yield TimeStep(index=i, time=time)
