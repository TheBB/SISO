"""Main Siso API definition.

This file provides the core types used in the data conversion pipeline. It
should import as litte as possible, and be imported almost everywhere else.

The most important classes defined here are:

- Basis: a specific discretization of the spatial domain
- Field: an evaluable quantity defined on a basis
- Step: a single instance of what is usually (but not always) the time axis
- Zone: a distinct and identified region of space
- Topology: the specific discretization for a basis in a zone
- Source: a data set which contains collections of the above five classes
  according to certain rules

Although this file provides a finished implementation for Zone, only abstract
definitions for the others are provided. Bare-bones implementations for those
can be found in the siso.impl module. They are usea as-is in some sources, or
extended in others.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import wraps
from typing import (
    TYPE_CHECKING,
    ClassVar,
    NewType,
    Protocol,
    Self,
    cast,
    overload,
)

import numpy as np
from attrs import Factory, asdict, define
from numpy import floating
from numpy.typing import NDArray

from siso.types import Float

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Sequence
    from pathlib import Path
    from types import TracebackType
    from typing import Any

    from siso.types import f32d, u32d
    from siso.util.field_data import FloatFieldData

    from .util.field_data import IntFieldData


Point = NewType("Point", tuple[Float, ...])
Points = NewType("Points", tuple[Point, ...])
KnotVector = NewType("KnotVector", NDArray[floating])
Knots = NewType("Knots", tuple[KnotVector, ...])


class Nodal: ...


class Cellular: ...


class Aritmetical(Protocol):
    def __add__(self, other: int, /) -> Self: ...

    def __sub__(self, other: int, /) -> Self: ...


class ShapeTuple[M: (Nodal, Cellular), P: Aritmetical](tuple[P, ...]):
    @overload
    def __new__(cls, *args: P) -> Self: ...

    @overload
    def __new__(cls, arg: Iterable[P]) -> Self: ...

    def __new__(cls, *args):  # type: ignore[no-untyped-def]
        if hasattr(args[0], "__iter__"):
            return super().__new__(cls, iter(args[0]))
        return super().__new__(cls, args)

    @property
    def pardim(self) -> int:
        return len(self)

    @property
    def nodal(self: ShapeTuple[Cellular, P]) -> ShapeTuple[Nodal, P]:
        return ShapeTuple(k + 1 for k in self)

    @property
    def cellular(self: ShapeTuple[Nodal, P]) -> ShapeTuple[Cellular, P]:
        return ShapeTuple(k - 1 for k in self)


NodeShape = ShapeTuple[Nodal, int]
CellShape = ShapeTuple[Cellular, int]


@define
class SisoError(Exception):
    msg: str

    def show(self) -> str:
        return self.msg


class Unsupported(SisoError): ...


class BadInput(SisoError): ...


class Unexpected(SisoError): ...


class ZoneShape(Enum):
    """The shape of a zone."""

    # 1D
    Line = auto()

    # 2D
    Triangle = auto()
    Quatrilateral = auto()

    # 3D
    Hexahedron = auto()

    # Misc
    Shapeless = auto()


@define(frozen=True)
class Zone[K]:
    """A zone is a distinct and identified region of the spatial domain.

    Most sources provide only one zone (a single grid for the entire domain),
    while some (such as IFEM) are multi-zoned: the spatial domain is divided
    into several distinct regions, each with their own distinct discretization.

    Attributes:
    - shape: the shape of the zone
    - coords: a sequence of points locating the corners of the zone
    - key: any object identfying the zone
    """

    shape: ZoneShape
    coords: Points
    key: K


class Endianness(Enum):
    """Enumeration for endianness options."""

    Native = "native"
    Little = "little"
    Big = "big"

    def make_dtype(self, root: str) -> np.dtype:
        """Create a numpy dtype with the appropriate endianness using `root`
        (e.g. 'u4', 'f4') as the numpy data type string identifier.
        """
        if self == Endianness.Native:
            return np.dtype(f"={root}")
        if self == Endianness.Little:
            return np.dtype(f"<{root}")
        return np.dtype(f">{root}")

    def u4_type(self) -> u32d:
        """Create a four-byte unsigned integer numpy dtype with the appropriate
        endianness.
        """
        return self.make_dtype("u4")

    def f4_type(self) -> f32d:
        """Create a four-byte floating point numpy dtype with the appropriate
        endianness.
        """
        return self.make_dtype("f4")


class Dimensionality(Enum):
    """Enumeration for dimensionality options."""

    Volumetric = "volumetric"
    Planar = "planer"
    Extrude = "extrude"

    def out_is_volumetric(self) -> bool:
        """Return true if this dimensionality is volumetric in essence (either
        volumetric or extrude).
        """
        return self != Dimensionality.Planar

    def in_allows_planar(self) -> bool:
        """Return true if this dimensionality allows planar fields as input
        (either planar or extrude).
        """
        return self != Dimensionality.Volumetric


class Staggering(Enum):
    Outer = "outer"
    Inner = "inner"


class Rationality(Enum):
    """Enumeration for options regarding how to resolve spline objects which may
    or may not be rational.
    """

    Always = "always"
    Never = "never"


@define
class SplitFieldSpec:
    """A specification for splitting a field into multiple components.

    These objects may be present in a source's properties, in which case the
    pipeline will apply a Split filter, which reads them and splits the fields
    accordingly.

    Attributes:
    - source_name: name of the input field from which the components are sourced
    - new_name: name of the new field
    - components: indices of the components to extract
    - destroy: if true, the source field will be suppressed from further processing
    - splittable: if true, the new fields will be marked as splittable
    """

    source_name: str
    new_name: str
    components: list[int]
    destroy: bool = True
    splittable: bool = False


@define
class RecombineFieldSpec:
    """A specification for recombining multiple fields into a new field.

    These object may be present in a source's properties, in which case the
    pipeline will apply a Recombine filter, which reads them and recombines
    the fields accordingly.

    Attributes:
    - source_names: list of input fields from which data is sourced
    - new_name: name of the new field
    """

    source_names: list[str]
    new_name: str


class StepInterpretation(Enum):
    """Enumeration for possible interpretations of the 'time' axis."""

    Time = auto()
    Eigenmode = auto()
    EigenFrequency = auto()

    @property
    def is_time(self) -> bool:
        """True if the 'time' axis should be interpreted as actual time."""
        return self == StepInterpretation.Time

    @property
    def is_eigen(self) -> bool:
        """True if the 'time' axis should be interpreted as a sequence of
        eigenmodes.
        """
        return self != StepInterpretation.Time

    def __str__(self) -> str:
        return {
            StepInterpretation.Time: "Time",
            StepInterpretation.Eigenmode: "Eigenvalue",
            StepInterpretation.EigenFrequency: "Frequency",
        }[self]


@define
class SourceProperties:
    """
    Object returned by the Source.properties attribute.

    This informs the pipeline about which filters to apply. It's crucial that
    the information here is correct, otherwise Siso will not function properly.

    Attributes:
    - instantaneous: true if the source can only contain one
        timestep.
    - globally_keyed: true if all the zone objects produced by the
        source has the global_key attribute set.
    - discrete_topology: true if all the topology objects produced by the source
        are subclasses of DiscreteTopology.
    - single_basis: true if the source can only contain one basis.
    - single_zoned: true if the source can only contain one zone.
    - step_interpretation: indicates how the pipeline should interpret the
        'time' axis.
    - split_fields: list of SplitFieldSpec objects indicating how the pipeline
        should split fields.
    - recombine_fields: list of RecombineFieldSpec objects indicating how the
        pipeline should recombine fields. Note that recombination happens after
        splitting, so that recombined fields may refer to split fields.

    Note that a source that CAN support multiple timesteps, bases or zones but
    where the input data just so happens to only have one should still report
    false for the respective attributes. That is, the values describe the
    theoretical capabilities of the TYPE, not the actual contents of the DATA.
    """

    instantaneous: bool
    globally_keyed: bool = False
    discrete_topology: bool = False
    single_basis: bool = False
    single_zoned: bool = False
    step_interpretation: StepInterpretation = StepInterpretation.Time

    split_fields: list[SplitFieldSpec] = Factory(list)
    recombine_fields: list[RecombineFieldSpec] = Factory(list)

    def update(self, **kwargs: Any) -> SourceProperties:
        """Construct a new SourceProperties object by selectively updating some
        attributes.
        """
        kwargs = {**asdict(self, recurse=False), **kwargs}
        return SourceProperties(**kwargs)


class ScalarInterpretation(Enum):
    """Enumeration for possible interpretations of a scalar field."""

    Generic = auto()
    Eigenmode = auto()

    def to_vector(self) -> VectorInterpretation:
        """Return the corresponding vector field interpretation (i.e. the
        interpretation of a one-component vector field whose component is this
        scalar).
        """
        if self == ScalarInterpretation.Eigenmode:
            return VectorInterpretation.Eigenmode
        return VectorInterpretation.Generic


class VectorInterpretation(Enum):
    """Enumeration for possible interpretations of a vector field."""

    Generic = auto()
    Displacement = auto()
    Eigenmode = auto()
    Flow = auto()

    def join(self, other: VectorInterpretation) -> VectorInterpretation:
        """Attempt to intepret the result of combining two vector fields in one
        (by merging their components).
        """
        if VectorInterpretation.Generic in (self, other):
            return VectorInterpretation.Generic
        assert self == other
        return self

    def to_scalar(self) -> ScalarInterpretation:
        """Return the corresponding interpretation of a single component of this
        vector field.
        """
        if self == VectorInterpretation.Eigenmode:
            return ScalarInterpretation.Eigenmode
        return ScalarInterpretation.Generic


class FieldType(Protocol):
    """Bundle of field metadata.

    Should be implemented by the classes Scalar, Vector and Geometry.
    """

    @property
    def num_comps(self) -> int:
        """Number of components in this field."""
        ...

    def as_scalar(self) -> FieldType:
        """Field type for a single component of this field."""
        ...

    def join(self, other: FieldType) -> FieldType:
        """Field type representing the component-wise joining of this field with
        another.
        """
        ...


@define(frozen=True)
class Scalar(FieldType):
    """Field metadata for scalar fields."""

    interpretation: ScalarInterpretation = ScalarInterpretation.Generic

    @property
    def num_comps(self) -> int:
        return 1

    def as_scalar(self) -> FieldType:
        return self

    def join(self, other: FieldType) -> FieldType:
        if isinstance(other, Scalar):
            # Joining two scalars: return a two-component vector type
            interpretation = self.interpretation.to_vector().join(other.interpretation.to_vector())
            return Vector(num_comps=2, interpretation=interpretation)
        # Joining a scalar to a vector: return an ncomps + 1 vector type
        assert isinstance(other, Vector)
        interpretation = self.interpretation.to_vector().join(other.interpretation)
        return Vector(num_comps=other.num_comps + 1, interpretation=interpretation)


@define(frozen=True)
class Vector(FieldType):
    """Field metadata for vector fields."""

    num_comps: int
    interpretation: VectorInterpretation = VectorInterpretation.Generic

    def as_scalar(self) -> FieldType:
        return Scalar(self.interpretation.to_scalar())

    def join(self, other: FieldType) -> FieldType:
        if isinstance(other, Scalar):
            # Joining a vector to a scalar: return a ncomps + 1 vector type
            interpretation = self.interpretation.join(other.interpretation.to_vector())
            return Vector(num_comps=self.num_comps + 1, interpretation=interpretation)
        # Joining a vector to a vector: return an m + n vector type
        assert isinstance(other, Vector)
        interpretation = self.interpretation.join(other.interpretation)
        return Vector(num_comps=self.num_comps + other.num_comps, interpretation=interpretation)

    def update(self, **kwargs: Any) -> Vector:
        """Construct a new vector type by partially updating some attributes."""
        kwargs = {**asdict(self, recurse=False), **kwargs}
        return Vector(**kwargs)


@define
class Geometry(FieldType):
    """Field metadata for geometries."""

    num_comps: int
    coords: CoordinateSystem

    def as_scalar(self) -> FieldType:
        # Geometry fields should not be sliced
        assert False

    def join(self, other: FieldType) -> FieldType:
        # Geometry fields should not be joined
        assert False

    def fits_system_name(self, name: str | None) -> bool:
        """Return true if the coordinate system name for this field is
        compatible with `name`.
        """
        if name is None:
            return True
        return self.coords.fits_system_name(name)


class CoordinateSystem(ABC):
    """Abstract base class for coordinate systems.

    Subclasses are implemented in the siso.coord module.

    Attributes:
    - name: name of the coordinate system (without parameters)
    """

    name: ClassVar[str]

    @classmethod
    @abstractmethod
    def make(cls, params: Sequence[str]) -> Self:
        """Construct a parametrized instance of this coordinate system from a
        sequence of string parameters.
        """
        ...

    @classmethod
    @abstractmethod
    def default(cls) -> Self:
        """Construct a parametrized instance of this coordinate system using
        default parameter values.
        """
        ...

    @property
    @abstractmethod
    def parameters(self) -> tuple[str, ...]:
        """Return a sequence of parameters as strings (used for stringifying)."""
        ...

    def fits_system_name(self, code: str) -> bool:
        """Return true if the name of this coordinate system fits `code`."""
        return code.casefold() == self.name.casefold()

    def __str__(self) -> str:
        params = ", ".join(p for p in self.parameters)
        return f"{self.name}({params})"


class Basis(ABC):
    """A basis represents a particular discretization of space.

    Sources may have one or more bases.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this basis."""
        ...


class Field(ABC):
    """A field represents an evaluable quantity in space and time."""

    @property
    @abstractmethod
    def cellwise(self) -> bool:
        """True if this field is cellwise constant. If so, the field data matrix
        should have number of rows equal to the number of cells of the
        corresponding topology."""
        ...

    @property
    @abstractmethod
    def splittable(self) -> bool:
        """True if this field can be split, either by the Split or Decompose
        filters.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of this field."""
        ...

    @property
    @abstractmethod
    def type(self) -> FieldType:
        """The type of this field."""
        ...

    @property
    def is_scalar(self) -> bool:
        """True if this field is scalar."""
        return isinstance(self.type, Scalar)

    @property
    def is_vector(self) -> bool:
        """True if this field is vector type (and not a geometry)."""
        return isinstance(self.type, Vector)

    @property
    def is_geometry(self) -> bool:
        """True if this field is a geometry."""
        return isinstance(self.type, Geometry)

    @property
    def is_eigenmode(self) -> bool:
        """True if this field should be interpreted as an eigenmode."""
        return (
            isinstance(self.type, Scalar)
            and self.type.interpretation == ScalarInterpretation.Eigenmode
            or isinstance(self.type, Vector)
            and self.type.interpretation == VectorInterpretation.Eigenmode
        )

    @property
    def is_displacement(self) -> bool:
        """True if this field should be interpreted as a displacement."""
        return isinstance(self.type, Vector) and self.type.interpretation == VectorInterpretation.Displacement

    @property
    def coords(self) -> CoordinateSystem:
        """Return the coordinate system of this field. Will crash for
        non-geometry fields.
        """
        return cast("Geometry", self.type).coords

    def fits_system_name(self, code: str) -> bool:
        """True if the coordinate system of this field is compatible with
        `code`. Will crash for non-geometry fields.
        """
        return isinstance(self.type, Geometry) and self.type.fits_system_name(code)

    @property
    def num_comps(self) -> int:
        """The number of components of this field."""
        return self.type.num_comps


class Step(Protocol):
    """A step represents an instance of the time axis (which may or may not be
    actual time)
    """

    @property
    def index(self) -> int:
        """Zero-based index of this step."""
        ...

    @property
    def value(self) -> Float | None:
        """The value associated with this step, if available. This will be time
        if the 'time axis' is actual time, or it may be eigenvalue if the 'time
        axis' is eigenmodes.
        """
        ...


@define
class ReaderSettings:
    """Settings passed to the Source.configure method. The source should honor
    these options as best as it is able.

    Attributes:
    - endianness: assume that the input has a specific endianness.
    - dimensionality: produce output with the required dimensionality.
    - staggering: ...
    - periodic: if true, stitch together periodic geometries.
    - mesh_filename: if the mesh is in a separate file from the input data,
        read it from this path.
    - rationality: if given, assume spline object with ambiguous rationality
        are either rational or non-rational, respectively.
    """

    endianness: Endianness
    dimensionality: Dimensionality
    staggering: Staggering
    periodic: bool
    mesh_filename: Path | None
    rationality: Rationality | None


class FieldDataFilter(Protocol):
    """Callable that modifies field data."""

    def __call__(self, field: Field, field_data: FloatFieldData, /) -> FloatFieldData: ...


class TopologyMerger(Protocol):
    """Callable that modifies a topology, and returns a new topology as well as
    a filter for converting corresponding field data.
    """

    def __call__(self, topology: Topology) -> tuple[Topology, FieldDataFilter]: ...


class Topology(ABC):
    """Abstract interface that must be implemented by topologies.

    A topology is a representation of a discretization local to a basis, step
    and zone.

    A topology is independent of fields, including geometry fields. Thus, in
    isolation, a topology does not 'know' where in space it is located, nor does
    a field 'know' how to interpret its data.
    """

    @property
    @abstractmethod
    def pardim(self) -> int:
        """Number of parametric dimensions."""
        ...

    @property
    @abstractmethod
    def num_nodes(self) -> int:
        """Number of nodes. Field data associated with this topology which
        aren't cellwise should have this many rows.
        """
        ...

    @property
    @abstractmethod
    def num_cells(self) -> int:
        """Number of cells. Filed data associated with this topology which
        are cellwise should have this many rows.
        """
        ...

    @abstractmethod
    def discretize(self, nvis: int) -> tuple[DiscreteTopology, FieldDataFilter]:
        """Return a discrete version of this topology, along with a field data
        filter that can be used to convert associated field data to be
        compatible with the discrete topology.
        """
        ...

    @abstractmethod
    def create_merger(self) -> TopologyMerger:
        """Return a topology merger object. This can be used to convert other
        compatible topologies at the same zone to a common 'merged' topology.
        This is used by the basis merge filter.
        """
        ...


class CellType(Enum):
    """Enumerates the supported grid cell types for discrete topologies."""

    Line = auto()
    Triangle = auto()
    Quadrilateral = auto()
    Hexahedron = auto()

    @property
    def is_tensor(self) -> bool:
        """Return true if this cell type is of tensor-product type (works in a
        structured topology.)
        """
        return True

    @property
    def pardim(self) -> int:
        """Return the number of parametric dimensions supported by this cell
        type.
        """
        if self in {CellType.Line}:
            return 1
        if self in {CellType.Quadrilateral}:
            return 2
        return 3


class CellOrdering(Enum):
    """Enumerates the supported cell ordering schemes."""

    Ifem = auto()
    Simra = auto()
    Siso = auto()
    Vtk = auto()


class DiscreteTopology(Topology):
    """A discrete topology consists of nodes connected by Lagriangian elements.

    This type of mesh is so common in many data formats that it requires special
    consideration.
    """

    @property
    @abstractmethod
    def celltype(self) -> CellType:
        """Return the cell type of this topology."""
        ...

    @property
    @abstractmethod
    def degree(self) -> int:
        """Return the polynomial degree of this topology."""
        ...

    @property
    @abstractmethod
    def cells(self) -> IntFieldData:
        """Return a matrix describing the cells (as nodal indices)."""
        ...

    @abstractmethod
    def cells_as(self, ordering: CellOrdering) -> IntFieldData:
        """Return a matrix describing the cells, ordered according to a certain
        convention.
        """
        ...


class Source[B: Basis, F: Field, S: Step, T: Topology, Z: Zone](ABC):
    """The primary object for representing a data source.

    This type is parametrized on the type of basis, field, step, topology and
    zone.

    The source object have methods that return bases, fields, steps and zones,
    as well as methods that consume these objects. It is the user's
    responsibility to ensure that any such object passed to a method of a source
    object was previously returned by a method of that same source object. In
    other words, don't mix and match such objects.

    Sources are allowed to assume, and user code must guarantee that:
    - timesteps are processed in turn,
    - within each timestep, fields are processed by basis, and
    - within each basis, the topology is processed before the field data.
    """

    def __enter__(self) -> Self:
        """Open the source data on disk."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close the source data on disk."""
        return

    @property
    @abstractmethod
    def properties(self) -> SourceProperties:
        """Return an object that describes the properties of this source.

        See SourceProperties for more information.
        """
        ...

    def configure(self, settings: ReaderSettings) -> None:
        """Advise the source to honor the provided settings as best as able.

        See ReaderSettings for more information.
        """
        return

    def use_geometry(self, geometry: Field) -> None:
        """Instruct the source of the geometry field that the pipeline intends
        to use.
        """
        return

    @abstractmethod
    def bases(self) -> Iterator[B]:
        """Return an iterator of all the bases available in the source.

        If self.properties.single_basis is true, this iterator has length
        one.
        """
        ...

    @abstractmethod
    def basis_of(self, field: Field) -> B:
        """Return the basis associated with a certain field."""
        ...

    @abstractmethod
    def fields(self, basis: Basis) -> Iterator[F]:
        """Return an iterator of all the non-geometry fields associated with a
        basis.
        """
        ...

    @abstractmethod
    def geometries(self, basis: Basis) -> Iterator[F]:
        """Return an iterator of all the geometry fields associated with a
        basis.
        """
        ...

    @abstractmethod
    def steps(self) -> Iterator[S]:
        """Return an iterator of all the steps in the source.

        If self.properties.instantaneous is true, this iterator has length
        one."""
        ...

    @abstractmethod
    def zones(self) -> Iterator[Z]:
        """Return an iterator of all the zones in the source.

        If self.properties.single_zoned is true, this iterator has length
        one. If self.properties.globally_keyed is true, the zone objects
        have the `global_key` attribute set.
        """
        ...

    @abstractmethod
    def topology(self, step: Step, basis: Basis, zone: Zone) -> T:
        """Return the topology associated with a step, basis and zone."""
        ...

    def topology_updates(self, step: Step, basis: Basis) -> bool:
        """Return true if the topologies associated with a given basis
        change at a given step. If false, the user may assume that the
        topologies from the previous timestep can be re-used.
        """
        return True

    @abstractmethod
    def field_data(self, step: Step, field: Field, zone: Zone) -> FloatFieldData:
        """Return the data of a field at a certain step and zone."""
        ...

    def field_updates(self, step: Step, field: Field) -> bool:
        """Return true if the data of a field changes at a given step.
        If false, the user may assume that the data from the previous timestep
        can be re-used."""
        return True

    def children(self) -> Iterator[Source]:
        """Return an iterator over all sub-sources for this source.

        This is useful for traversing a tree of 'filters": sources whose input
        data are other sources.
        """
        return
        yield

    def cast_globally_keyed(self) -> Source[B, F, S, T, Zone[int]]:
        """Cast the Z type to Zone[int] based on runtime check."""
        if not self.properties.globally_keyed:
            raise Unexpected("Source is not globally keyed")
        return cast("Source[B, F, S, T, Zone[int]]", self)

    def cast_discrete_topology(self) -> Source[B, F, S, DiscreteTopology, Z]:
        """Cast the T type to DiscreteTopology based on runtime check."""
        if not self.properties.discrete_topology:
            raise Unexpected("Source does not guarantee discrete topologies")
        return cast("Source[B, F, S, DiscreteTopology, Z]", self)

    def single_basis(self) -> B:
        """Singleton version of `basis()`."""
        if not self.properties.single_basis:
            raise Unexpected("Source does not guarantee single basis")
        return next(self.bases())

    def single_zone(self) -> Z:
        """Singleton version of `zone()`."""
        if not self.properties.single_zoned:
            raise Unexpected("Source does not guarantee single zone")
        return next(self.zones())

    def single_step(self) -> S:
        """Singleton version of `steps()`."""
        if not self.properties.instantaneous:
            raise Unexpected("Source does not guarantee single step")
        return next(self.steps())


def impl_basis_of[C, F: Field, B: Basis](
    func: Callable[[C, F], B],
) -> Callable[[C, Field], B]:
    @wraps(func)
    def inner(self: C, field: Field) -> B:
        return func(self, cast("F", field))

    return inner


def impl_fields[C, B: Basis, F: Field](
    func: Callable[[C, B], Iterator[F]],
) -> Callable[[C, Basis], Iterator[F]]:
    @wraps(func)
    def inner(self: C, basis: Basis) -> Iterator[F]:
        return func(self, cast("B", basis))

    return inner


def impl_field_data[C, F: Field, S: Step, Z: Zone](
    func: Callable[[C, S, F, Z], FloatFieldData],
) -> Callable[[C, Step, Field, Zone], FloatFieldData]:
    @wraps(func)
    def inner(self: C, step: Step, field: Field, zone: Zone) -> FloatFieldData:
        return func(self, cast("S", step), cast("F", field), cast("Z", zone))

    return inner


def impl_field_updates[C, F: Field, S: Step](
    func: Callable[[C, S, F], bool],
) -> Callable[[C, Step, Field], bool]:
    @wraps(func)
    def inner(self: C, step: Step, field: Field) -> bool:
        return func(self, cast("S", step), cast("F", field))

    return inner


def impl_geometries[C, B: Basis, F: Field](
    func: Callable[[C, B], Iterator[F]],
) -> Callable[[C, Basis], Iterator[F]]:
    @wraps(func)
    def inner(self: C, basis: Basis) -> Iterator[F]:
        return func(self, cast("B", basis))

    return inner


def impl_topology[C, B: Basis, S: Step, Z: Zone, T: Topology](
    func: Callable[[C, S, B, Z], T],
) -> Callable[[C, Step, Basis, Zone], T]:
    @wraps(func)
    def inner(self: C, step: Step, basis: Basis, zone: Zone) -> T:
        return func(self, cast("S", step), cast("B", basis), cast("Z", zone))

    return inner


def impl_topology_updates[C, B: Basis, S: Step](
    func: Callable[[C, S, B], bool],
) -> Callable[[C, Step, Basis], bool]:
    @wraps(func)
    def inner(self: C, step: Step, basis: Basis) -> bool:
        return func(self, cast("S", step), cast("B", basis))

    return inner


def impl_use_geometry[C, F: Field](
    func: Callable[[C, F], None],
) -> Callable[[C, Field], None]:
    @wraps(func)
    def inner(self: C, field: Field) -> None:
        return func(self, cast("F", field))

    return inner
