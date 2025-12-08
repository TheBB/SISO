"""This module defines two readers for related IFEM .hdf5 files: normal output
and eigenmode output.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import lru_cache
from itertools import chain, count, repeat
from typing import TYPE_CHECKING, ClassVar, Self, cast

import h5py
from attrs import define, field

from siso import api, util
from siso.api import (
    Topology,
    Zone,
    ZoneShape,
    impl_basis_of,
    impl_field_data,
    impl_field_updates,
    impl_fields,
    impl_geometries,
    impl_topology,
    impl_topology_updates,
    impl_use_geometry,
)
from siso.coord import Named
from siso.impl import Basis, Field, Step
from siso.topology import LrTopology, SplineTopology, UnstructuredTopology
from siso.util import FieldData

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from types import TracebackType

    from siso.types import IntVector
    from siso.util.field_data import FloatFieldData


class InvalidNcompsError(Exception):
    pass


@define(frozen=True)
class ZoneKey:
    """Key type used for IFEM zones."""

    name: str  # Name of basis
    step: int  # Timestep index
    patch: int  # Patch ID
    subblock: int  # Subblock index (in cases where patches contain multiple blocks)


@define(frozen=True)
class IfemPatch:
    """Used as the return type of `IfemBasis.patch_at`."""

    zone: Zone[ZoneKey]
    topology: api.Topology
    data: FloatFieldData
    node_mapping: IntVector | None
    cell_mapping: IntVector | None
    original_ndofs: int
    original_ncells: int


class Locator(ABC):
    """Abstract interface for locating objects in an IFEM HDF5 file.

    This is useful since normal HDF5 files have a slightly different layout to
    eigenmode HDF5 files.
    """

    @abstractmethod
    def patch_root(self, name: str, step: int) -> str:
        """Return the group path for patches belonging to a basis at a certain
        step.
        """
        ...

    def patch_path(self, name: str, step: int, patch: int) -> str:
        """Return the full path to a patch belonging to a basis at a certain
        step.
        """
        return f"{self.patch_root(name, step)}/{patch + 1}"

    @abstractmethod
    def coeff_root(self, basis_name: str, field_name: str, step: int, cellwise: bool) -> str:
        """Return the group path for coefficients belonging to a field under a
        certain basis at a given step.
        """
        ...

    def coeff_path(self, basis_name: str, field_name: str, step: int, patch: int, cellwise: bool) -> str:
        """Return the full path to a coefficient data set belonging to a field
        under a certain basis at a given step.
        """
        return f"{self.coeff_root(basis_name, field_name, step, cellwise)}/{patch + 1}"


class StandardLocator(Locator):
    """Locator interface used for standard IFEM HDF5 files."""

    def patch_root(self, name: str, step: int) -> str:
        return f"{step}/{name}/basis"

    def coeff_root(self, basis_name: str, field_name: str, step: int, cellwise: bool) -> str:
        subdir = "knotspan" if cellwise else "fields"
        return f"{int(step)}/{basis_name}/{subdir}/{field_name}"


class EigenLocator(Locator):
    """Locator interface used for eigenmode IFEM HDF5 files."""

    def patch_root(self, name: str, step: int) -> str:
        return f"0/{name}/basis"

    def coeff_root(self, basis_name: str, field_name: str, step: int, cellwise: bool) -> str:
        return f"0/{basis_name}/Eigenmode/{step + 1}"


def is_legal_group_name(name: str) -> bool:
    """To discriminate legal IFEM HDF5 files from other HDF5 files, all groups
    at top level must satisfy this test.
    """

    # IFEM top-level groups are all numbered, except possibly "anasol" and
    # "log".
    try:
        int(name)
        return True
    except ValueError:
        return name.casefold() in ("anasol", "log")


@define(frozen=True)
class IfemBasis(Basis):
    """Implementation of the Basis interface used for IFEM readers."""

    # Locator for finding things in the HDF5 file.
    locator: Locator = field(eq=False, repr=False)

    # List of fields belonging to this basis (populated by the Field class).
    fields: list[IfemField] = field(factory=list, init=False, eq=False, repr=False)

    @lru_cache(maxsize=1)
    def num_patches(self, source: Ifem) -> int:
        """Return the number of patches in this basis."""
        i = 0
        for i in count():
            if self.patch_path(0, i) not in source.h5:
                break
        return i

    def updates_at(self, step: int, source: Ifem) -> bool:
        """Return true if this basis changes at the given timestep."""
        return self.locator.patch_root(self.name, step) in source.h5

    def last_update_before(self, step: int, source: Ifem) -> int:
        """Return the most recent timestep prior to `step` where this basis was
        updated.
        """
        while not self.updates_at(step, source):
            step -= 1
        return step

    def patch_path(self, step: int, patch: int) -> str:
        """Return the full path to a patch in the HDF5 file."""
        return self.locator.patch_path(self.name, step, patch)

    @lru_cache(maxsize=128)
    def patches_at(
        self,
        step: int,
        patch: int,
        source: Ifem,
    ) -> tuple[IfemPatch, ...]:
        """Obtain the patch at the given step by index. This returns a tuple
        with three items: zone, topology and field data.
        """

        # Find the most recent update prior to the requested timestep.
        step = self.last_update_before(step, source)

        # Patches are stored as bytestrings, to be interpreted by a topology
        # class. We discriminate based on the first few bytes.
        patchdata = source.h5[self.patch_path(step, patch)][:]
        initial = patchdata[:20].tobytes()
        raw_data = memoryview(cast("bytes", patchdata)).tobytes()
        topology: api.Topology

        cps: FloatFieldData
        if initial.startswith(b"# LAGRANGIAN"):
            # Lagrangian IFEM meshes may consist of multiple blocks. In case
            # there are e.g. mixed quads and triangles, we get one block for
            # each cell type. In this case, we return multiple zones.
            retval: list[IfemPatch] = []
            for i, (
                corners,
                topology,
                cps,
                node_mapping,
                cell_mapping,
                original_ndofs,
                original_ncells,
            ) in enumerate(UnstructuredTopology.from_ifem(raw_data)):
                zone = Zone(shape=ZoneShape.Shapeless, coords=corners, key=ZoneKey(self.name, step, patch, i))
                retval.append(
                    IfemPatch(
                        zone, topology, cps, node_mapping, cell_mapping, original_ndofs, original_ncells
                    )
                )
            return tuple(retval)

        if initial.startswith(b"# LRSPLINE"):
            corners, topology, cps = next(LrTopology.from_bytes(raw_data, source.rationality))
        else:
            # Shame! GoTools files don't have 'magic bytes' at the beginning.
            corners, topology, cps = next(SplineTopology.from_bytes(raw_data))

        # Spline patches never have irregular shapes.
        shape = [ZoneShape.Line, ZoneShape.Quatrilateral, ZoneShape.Hexahedron][topology.pardim - 1]

        zone = Zone(shape=shape, coords=corners, key=ZoneKey(self.name, step, patch, 0))
        return (IfemPatch(zone, topology, cps, None, None, topology.num_nodes, topology.num_cells),)

    @lru_cache(maxsize=1)
    def ncomps(self, source: Ifem) -> int:
        """Calculate the number of components (physical dimensionality) of the
        basis.
        """
        patch = self.patches_at(0, 0, source)[0]
        return patch.data.num_comps


@define(frozen=True)
class IfemField(Field):
    """Implementation of the Field interface used for IFEM readers."""

    # The basis to which this field belongs.
    basis: IfemBasis = field(kw_only=True, eq=False)

    def splits(self) -> Iterator[api.SplitFieldSpec]:
        """Suggest splitting this field into components.

        IFEM occasionally produces fields in 'merged' form, with names separated
        by '&&", e.g. a three-component field called 'a&&b&&c' where a, b, c are
        not otherwise related. Such fields should be split in their respective
        components before output - the combined field is not physically relevant
        to anything.

        The `SplitFieldSpec` objects produced by this method are passed on to
        the data processing pipeline through `SourceProperties` and then to
        `Split` filter, where the actual splitting happens.
        """

        if not self.splittable or "&&" not in self.name:
            return

        # Split on '&&'
        component_names = [comp.strip() for comp in self.name.split("&&")]

        # If there's a space in the first component name, that's a prefix that
        # should be added to all the component names.
        if " " in component_names[0]:
            prefix, component_names[0] = component_names[0].split(" ", maxsplit=1)
            component_names = [f"{prefix} {comp}" for comp in component_names]

        # Yield instructions for splitting the field
        for i, comp in enumerate(component_names):
            yield api.SplitFieldSpec(
                source_name=self.name,
                new_name=comp,
                components=[i],
                destroy=True,
            )

    @abstractmethod
    def cps_at(self, step: int, patch: int, block: int, source: Ifem) -> FloatFieldData:
        """Return the control point data for this field at a given step and patch."""
        ...

    @abstractmethod
    def updates_at(self, step: int, source: Ifem) -> bool:
        """Return true if this field has an update at the given step."""
        ...


class IfemGeometryField(IfemField):
    """Implementation of IfemField for geometry fields.

    Geometry fields are different from other fields, since their field data
    (control points) come directly from the basis, and not from an array data
    set in the HDF5 file.
    """

    def __init__(self, basis: IfemBasis, source: Ifem):
        super().__init__(
            # Geometry fields have names that are identical with their basis name
            name=basis.name,
            # The coordinate system has a 'name' (same as the basis name), but is otherwise unspecified
            type=api.Geometry(basis.ncomps(source), coords=Named(basis.name)),
            splittable=False,
            basis=basis,
        )

    def cps_at(self, step: int, patch: int, block: int, source: Ifem) -> FloatFieldData:
        p = self.basis.patches_at(step, patch, source)[block]
        if p.node_mapping is not None:
            return p.data.slice_dofs(p.node_mapping)
        return p.data

    def updates_at(self, step: int, source: Ifem) -> bool:
        return self.basis.updates_at(step, source)


class IfemStandardField(IfemField):
    """Full implementation of IfemField for standard (non-geometry) fields."""

    def __init__(
        self,
        name: str,
        cellwise: bool,
        basis: IfemBasis,
        source: Ifem,
    ):
        # Calculate the number of components. We would like to just use
        # `self.cps_at` to get the control points, but that won't work until
        # this class is initialized, which requires the field type, which we
        # can't construct until we know the number of components. We can't
        # mutate the class post-initialization either, because it's frozen.
        # Oh well.

        # Find a patch to get the number of coefficients from
        root_path = basis.locator.coeff_root(basis.name, name, 0, cellwise)
        patch_id = int(next(iter(source.h5[root_path].keys()))) - 1  # type: ignore[attr-defined]
        coeff_path = basis.locator.coeff_path(basis.name, name, 0, patch_id, cellwise)
        cps = source.h5[coeff_path][:]

        p = basis.patches_at(0, patch_id, source)[0]
        cps = cps.reshape(p.original_ncells if cellwise else p.original_ndofs, -1)
        if not cellwise and p.node_mapping is not None:
            cps = cps[p.node_mapping, ...]
        elif cellwise and p.cell_mapping is not None:
            cps = cps[p.cell_mapping, ...]
        cps = cps.flatten()

        divisor = p.topology.num_cells if cellwise else p.topology.num_nodes

        ncomps, remainder = divmod(len(cps), divisor)
        if remainder != 0:
            raise InvalidNcompsError(
                f"Field '{name}': inconsistent number of components: "
                f"{len(cps)}/{divisor} = {ncomps} (remainder {remainder})"
            )

        # Construct the field type based on the number of components. We get the
        # default interpretation from the source object (generally either
        # 'generic' for standard HDF5 files or 'eigenmode' for eigenmode HDF5
        # files).
        tp: api.FieldType
        if ncomps > 1:
            tp = api.Vector(num_comps=ncomps, interpretation=source.default_vector)
        else:
            tp = api.Scalar(interpretation=source.default_scalar)

        super().__init__(name, type=tp, cellwise=cellwise, basis=basis)
        basis.fields.append(self)

    def __repr__(self) -> str:
        return f"Field({self.name}, {'cellwise' if self.cellwise else 'nodal'}, {self.basis.name})"

    def updates_at(self, step: int, source: Ifem) -> bool:
        return self.coeff_root(step) in source.h5

    def last_update_before(self, step: int, source: Ifem) -> int:
        """Return the most recent step prior to the given step where this field
        had an update.
        """
        while not self.updates_at(step, source):
            step -= 1
        return step

    def coeff_root(self, step: int) -> str:
        """Return the group path for coefficients belonging to this field
        at a given step.
        """
        return self.basis.locator.coeff_root(
            self.basis.name,
            self.name,
            step,
            self.cellwise,
        )

    def coeff_path(self, step: int, patch: int) -> str:
        """Return the full path for coefficients belonging to this field
        at a given step and patch.
        """
        return self.basis.locator.coeff_path(
            self.basis.name,
            self.name,
            step,
            patch,
            self.cellwise,
        )

    def cps_at(self, step: int, patch: int, block: int, source: Ifem) -> FloatFieldData:
        """Return the control points for this field at a given step and
        patch.
        """
        p = self.basis.patches_at(step, patch, source)[block]
        cps = source.h5[self.coeff_path(step, patch)][:]
        cps = cps.reshape(-1, self.num_comps)

        if not self.cellwise and p.node_mapping is not None:
            cps = cps[p.node_mapping, ...]
        elif self.cellwise and p.cell_mapping is not None:
            cps = cps[p.cell_mapping, ...]

        return FieldData(cps)


class Ifem(api.Source[IfemBasis, IfemField, Step, Topology, Zone]):
    """Source class for standard (non-eigenmode) IFEM HDF5 files."""

    filename: Path

    # Populated by __enter__
    h5: h5py.File

    # Populated by use_geometry - user's responsibility to call that method
    geometry: IfemBasis

    # Populated by discover_bases and discover_fields (called by __enter__)
    _bases: dict[str, IfemBasis]
    _fields: dict[str, IfemField]

    # Options that can be overridden by subclasses (for eigenmodes)
    locator: ClassVar[Locator] = StandardLocator()
    default_scalar: ClassVar[api.ScalarInterpretation] = api.ScalarInterpretation.Generic
    default_vector: ClassVar[api.VectorInterpretation] = api.VectorInterpretation.Generic

    # LR-Splines don't natively support rationals. We try to deterime which
    # splines are rational based on the number of components, but this isn't
    # always possible. This setting can be set in the CLI to override this.
    rationality: api.Rationality | None = None

    @staticmethod
    def applicable(path: Path) -> bool:
        """Return true if the given path is (probably) a valid IFEM HDF5 file."""
        try:
            with h5py.File(path, "r") as f:
                assert all(is_legal_group_name(name) for name in f)
            return True
        except (AssertionError, OSError):
            return False

    def __init__(self, filename: Path):
        self.filename = filename
        self._fields = {}

    def __enter__(self) -> Self:
        self.h5 = h5py.File(self.filename, "r").__enter__()

        # Find all bases
        self.discover_bases()
        for basis in self._bases.values():
            logging.debug(
                f"Basis {basis.name} with {util.pluralize(basis.num_patches(self), 'patch', 'patches')}"
            )

        # Find all fields
        self.discover_fields()

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.h5.__exit__(exc_type, exc_val, exc_tb)

    @property
    def properties(self) -> api.SourceProperties:
        # Collect field splitting and recombination suggestions to pass on to
        # the pipeline and the Split/Recombine filters.
        splits, recombineations = self.propose_recombinations()
        return api.SourceProperties(
            instantaneous=False,
            split_fields=splits,
            recombine_fields=recombineations,
        )

    def configure(self, settings: api.ReaderSettings) -> None:
        self.rationality = settings.rationality

    @property
    def num_steps(self) -> int:
        """Number of steps."""
        return len(set(self.h5) - {"anasol", "log"})

    def step_groups(self) -> Iterator[h5py.Group]:
        """Iterator over all groups for steps in the HDF5 file."""
        for index in range(self.num_steps):
            yield self.h5[str(index)]

    def make_basis(self, name: str) -> IfemBasis:
        """Create a basis with a given name."""
        return IfemBasis(name, self.locator)

    def discover_bases(self) -> None:
        """Discover all the bases in the HDF5 file."""

        # We use a dictionary here to preserve order (for testing and
        # reproducibility purposes). In principle this is used as a set. We
        # iterate through all the top-level groups in the HDF5 file and collect
        # the second-level group names, minus known non-basis entries.
        basis_names = dict.fromkeys(
            name for name in chain.from_iterable(self.h5.values()) if name != "timeinfo"
        )

        # Construct bases and populate the _basis dict. Eliminate bases that
        # don't have patches.
        bases = (self.make_basis(name) for name in basis_names)
        self._bases = {basis.name: basis for basis in bases if basis.num_patches(self) > 0}

    def discover_fields(self) -> None:
        """Discover all the fields in the HDF5 file."""

        # Keep track of fields that have issued warnings, so we don't
        # show multiple
        warnings: set[str] = set()

        # Iterate over all steps and bases
        for step_grp in self.step_groups():
            for basis_name, basis_grp in step_grp.items():
                if basis_name not in self._bases:
                    continue

                # Collect all the fields seen in this group. This is a sequence
                # of (name, bool) tuples, where the second element is true if
                # this field is cellwise. The cellwise fields are in the
                # 'knotspan' group, while the nodal fields are in the 'fields'
                # group.
                fields: Iterator[tuple[str, bool]] = chain(
                    zip(basis_grp.get("fields", ()), repeat(False)),
                    zip(basis_grp.get("knotspan", ()), repeat(True)),
                )

                # Construct new field objects for the fields we haven't seen
                # before.
                for field_name, cellwise in fields:
                    if field_name in self._fields or field_name in warnings:
                        continue
                    try:
                        field = IfemStandardField(
                            name=field_name,
                            cellwise=cellwise,
                            basis=self._bases[basis_name],
                            source=self,
                        )
                    except InvalidNcompsError as exc:
                        logging.warning(exc)
                        logging.warning("This field will be skipped")
                        warnings.add(field_name)
                        continue
                    self._fields[field_name] = field

    @lru_cache(maxsize=1)
    def propose_recombinations(self) -> tuple[list[api.SplitFieldSpec], list[api.RecombineFieldSpec]]:
        """Collect suggested field splits and recombinations.

        This method, which is expensive to run, is called from properties, which
        should be cheap. Therefore memoize the return value.
        """

        # Each field proposes their own splits.
        splits = list(chain.from_iterable(field.splits() for field in self._fields.values()))

        # Keep a list of candidates for potential recombination.
        candidates: dict[str, list[str]] = defaultdict(list)

        # We can recombine both newly split fields as well as already existing ones.
        field_names = chain(self._fields, (split.new_name for split in splits))

        # A field is a candidate for recombination if it is suffixed '_x', '_y'
        # or '_z', in which case the prefix is the suggested name for the
        # recombined field.
        for field_name in field_names:
            if len(field_name) <= 2 or field_name[-2] != "_":
                continue
            prefix, suffix = field_name[:-2], field_name[-1]
            if suffix not in "xyz":
                continue
            candidates[prefix].append(field_name)

        # Finalize recombinations and return.
        recombinations = [
            api.RecombineFieldSpec(source_names, new_name)
            for new_name, source_names in candidates.items()
            if new_name not in self._fields and len(source_names) > 1
        ]

        return splits, recombinations

    @impl_use_geometry
    def use_geometry(self, geometry: Field) -> None:
        self.geometry = self._bases[geometry.name]

    def steps(self) -> Iterator[Step]:
        for i, group in enumerate(self.step_groups()):
            time = group["timeinfo/level"][0] if "timeinfo/level" in group else float(i)
            yield Step(index=i, value=time)

    def zones(self) -> Iterator[Zone]:
        for patch_id in range(self.geometry.num_patches(self)):
            for patch in self.geometry.patches_at(0, patch_id, self):
                yield patch.zone

    def bases(self) -> Iterator[IfemBasis]:
        return iter(self._bases.values())

    @impl_basis_of
    def basis_of(self, field: IfemField) -> IfemBasis:
        return field.basis

    @impl_geometries
    def geometries(self, basis: IfemBasis) -> Iterator[IfemField]:
        yield IfemGeometryField(basis, self)

    @impl_fields
    def fields(self, basis: IfemBasis) -> Iterator[IfemField]:
        return iter(basis.fields)

    @impl_topology
    def topology(self, step: Step, basis: IfemBasis, zone: Zone[ZoneKey]) -> api.Topology:
        block = basis.patches_at(step.index, zone.key.patch, self)[zone.key.subblock]
        return block.topology

    @impl_topology_updates
    def topology_updates(self, step: Step, basis: IfemBasis) -> bool:
        return basis.updates_at(step.index, self)

    @impl_field_data
    def field_data(self, step: Step, field: IfemField, zone: Zone[ZoneKey]) -> FloatFieldData:
        return field.cps_at(step.index, zone.key.patch, zone.key.subblock, self)

    @impl_field_updates
    def field_updates(self, step: Step, field: IfemField) -> bool:
        if field.is_geometry:
            return field.basis.updates_at(step.index, self)
        return field.updates_at(step.index, self)


class IfemModes(Ifem):
    """Source class for standard (non-eigenmode) IFEM HDF5 files."""

    # Eigenmode HDF5 files are slightly different, but most of the machinery
    # will work so long as these objects are changed.
    locator = EigenLocator()
    default_scalar = api.ScalarInterpretation.Eigenmode
    default_vector = api.VectorInterpretation.Eigenmode

    @staticmethod
    def applicable(path: Path) -> bool:
        """Return true if the given path is (probably) a valid IFEM HDF5
        eigenmode file.
        """
        try:
            with h5py.File(path, "r") as f:
                assert "0" in f
                basis_name = next(iter(f["0"]))
                assert "Eigenmode" in f[f"0/{basis_name}"]
            return True
        except (AssertionError, OSError):
            return False

    @property
    def properties(self) -> api.SourceProperties:
        # Check whether the eigenmodes are of frequency type or not, then report
        # the appropriate step interpretation.
        basis = next(iter(self._bases.values()))
        group = self.h5[f"0/{basis.name}/Eigenmode/1"]
        if "Value" in group:
            step = api.StepInterpretation.Eigenmode
        else:
            assert "Frequency" in group
            step = api.StepInterpretation.EigenFrequency
        return super().properties.update(step_interpretation=step)

    def discover_bases(self) -> None:
        # Eigenmode files only have one basis.
        group = self.h5["0"]
        basis_name = util.only(group)
        self._bases = {basis_name: self.make_basis(basis_name)}

    def discover_fields(self) -> None:
        # Eigenmode files only have one field.
        basis = util.only(self._bases.values())
        self._fields = {
            "Mode Shape": IfemStandardField("Mode Shape", cellwise=False, basis=basis, source=self)
        }

    @property
    def num_steps(self) -> int:
        basis = util.only(self._bases.values())
        return len(self.h5[f"0/{basis.name}/Eigenmode"])

    def step_groups(self) -> Iterator[h5py.Group]:
        basis = util.only(self._bases.values())
        for index in range(self.num_steps):
            yield self.h5[f"0/{basis.name}/Eigenmode/{index + 1}"]

    def steps(self) -> Iterator[Step]:
        for i, group in enumerate(self.step_groups()):
            time = group["Value"][0] if "Value" in group else group["Frequency"][0]
            yield Step(index=i, value=time)

    @impl_field_updates
    def field_updates(self, timestep: Step, field: IfemField) -> bool:
        if field.is_geometry:
            return timestep.index == 0
        return super().field_updates(timestep, field)
