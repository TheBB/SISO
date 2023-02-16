from __future__ import annotations

from functools import lru_cache
from itertools import chain, repeat
import logging
from pathlib import Path

import h5py

from ..api import Source, SourceProperties, SplitFieldSpec, RecombineFieldSpec, Topology
from ..field import Field, FieldType, FieldData
from ..topology import SplineTopology
from ..zone import Zone, Shape
from .. import util

from typing_extensions import Self
from typing import (
    Dict,
    Iterator,
    Optional,
    List,
    Set,
    Tuple,
)


def is_legal_group_name(name: str) -> bool:
    try:
        int(name)
        return True
    except ValueError:
        return name.lower() in ('anasol', 'log')


class IfemBasis:
    name: str
    update_steps: Set[int]
    num_patches: int

    def __init__(self, name: str, steps: Iterator[h5py.Group]):
        self.name = name
        self.update_steps = set()
        self.num_patches = 0

        subpath = f'{name}/basis'
        for i, group in enumerate(steps):
            if subpath not in group:
                continue
            self.update_steps.add(i)
            self.num_patches = max(self.num_patches, len(group[subpath]))

    def __repr__(self) -> str:
        return f'Basis({self.name}, updates={len(self.update_steps)}, num_patches={self.num_patches})'

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: IfemBasis) -> bool:
        return self.name == other.name

    def group_path(self, step: int) -> str:
        return f'{int(step)}/{self.name}/basis'

    def patch_path(self, step: int, patch: int) -> str:
        return f'{self.group_path(step)}/{patch+1}'

    @lru_cache(maxsize=8)
    def patch_at(self, step: int, patch: int, source: Ifem) -> Tuple[Zone, Topology, FieldData]:
        while step not in self.update_steps:
            step -= 1

        patchdata = source.h5[self.patch_path(step, patch)][:]
        # initial = patchdata[:20].tobytes()
        raw_data = memoryview(patchdata).tobytes()
        # if initial.startswith(b'# LAGRANGIAN'):
        #     # topo, nodes = UnstructuredTopology.from_lagrangian(g2bytes)
        #     pass
        # elif initial.startswith(b'# LRSPLINE'):
        #     # topo, nodes = next(LRTopology.from_string(g2bytes.read()))
        #     pass
        # else:
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

    def splits(self) -> SplitFieldSpec:
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

    def group_path(self, step: int) -> str:
        subdir = 'knotspan' if self.cellwise else 'fields'
        return f'{int(step)}/{self.basis.name}/{subdir}/{self.name}'

    def patch_path(self, step: int, patch: int) -> str:
        return f'{self.group_path(step)}/{patch+1}'

    @lru_cache(maxsize=1)
    def ncomps(self, source: Ifem) -> int:
        _, _, basis_cps = self.basis.patch_at(0, 0, source)
        my_cps = self.raw_cps_at(0, 0, source)
        assert len(my_cps) % basis_cps.ndofs == 0
        return len(my_cps) // basis_cps.ndofs

    @lru_cache(maxsize=8)
    def raw_cps_at(self, step: int, patch: int, source: Ifem) -> FieldData:
        return source.h5[self.patch_path(step, patch)][:]


class Ifem(Source):
    filename: Path
    h5: h5py.File

    _bases: Dict[str, IfemBasis]
    _fields: Dict[str, IfemField]

    @staticmethod
    def applicable(path: Path) -> bool:
        try:
            with h5py.File(path, 'r') as f:
                assert all(is_legal_group_name(name) for name in f)
            return True
        except:
            return False

    def __init__(self, filename: Path):
        self.filename = filename
        self._fields = {}

    def __enter__(self) -> Self:
        self.h5 = h5py.File(self.filename, 'r').__enter__()
        self.discover_bases()
        self.discover_fields()
        return self

    def __exit__(self, *args):
        self.h5.__exit__(*args)

    @property
    @lru_cache(maxsize=1)
    def properties(self) -> SourceProperties:
        splits, recombineations = self.propose_recombinations()
        return SourceProperties(
            instantaneous=False,
            split_fields=splits,
            recombine_fields=recombineations,
        )

    @property
    def nsteps(self) -> int:
        return len(self.h5)

    def timestep_groups(self) -> Iterator[h5py.Group]:
        for index in range(self.nsteps):
            yield self.h5[str(index)]

    def discover_bases(self):
        basis_names = set(chain.from_iterable(self.h5.values())) - {'timeinfo'}
        bases = (IfemBasis(name, self.timestep_groups()) for name in basis_names)
        self._bases = {
            basis.name: basis
            for basis in bases
            if basis.update_steps and basis.num_patches > 0
        }

        for basis in self._bases.values():
            logging.debug(
                f"Basis {basis.name} updates at "
                f"{util.pluralize(len(basis.update_steps), 'step', 'steps')} "
                f"with {util.pluralize(basis.num_patches, 'patch', 'patches')}"
            )

    def discover_fields(self):
        for step_grp in self.timestep_groups():
            for basis_name, basis_grp in step_grp.items():
                if basis_name not in self._bases:
                    continue

                fields: Iterator[str, bool] = chain(
                    zip(basis_grp.get('fields', ()), repeat(False)),
                    zip(basis_grp.get('knotspan', ()), repeat(True)),
                )
                for field_name, cellwise in fields:
                    self._fields[field_name] = IfemField(
                        name=field_name,
                        cellwise=cellwise,
                        basis=self._bases[basis_name],
                    )

    def propose_recombinations(self) -> Tuple[List[SplitFieldSpec], List[RecombineFieldSpec]]:
        splits = list(chain.from_iterable(field.splits() for field in self._fields.values()))
        return (splits, [])

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
                type=FieldType.Generic,
                ncomps=field.ncomps(self),
                cellwise=field.cellwise,
            )

        # for field in self._fields.values():



        #             for subfield in superfield.split():
        #                 self._fields[subfield.name] = subfield

        # candidates: Dict[str, List[str]] = {}
        # for field_name in self._fields:
        #     if len(field_name) <= 2 or field_name[-2] != '_':
        #         continue
        #     prefix, suffix = field_name[:-2], field_name[-1]
        #     if prefix in self._fields or suffix not in 'xyz':
        #         continue
        #     candidates.setdefault(prefix, []).append(field_name)

        # for super_name, sub_names in candidates.items():
        #     if not (1 < len(sub_names) < 4):
        #         continue
        #     sub_names = sorted(sub_names, key=lambda s: s[-1])
#


        # to_add: List[IfemField] = []
        # to_delete: List[IfemField] = []
        # for field_name in self._fields:

