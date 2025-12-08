from __future__ import annotations

from enum import Enum, auto
from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from netCDF4 import Dataset
from scipy.spatial.transform import Rotation

from siso import api, util
from siso.api import (
    CellShape,
    NodeShape,
    Zone,
    ZoneShape,
    impl_basis_of,
    impl_field_data,
    impl_fields,
    impl_geometries,
    impl_topology,
    impl_topology_updates,
    impl_use_geometry,
)
from siso.coord import Generic, Geodetic, Wgs84
from siso.impl import Basis, Field, Step
from siso.topology import CellType, DiscreteTopology, StructuredTopology, UnstructuredTopology
from siso.util import FieldData

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from types import TracebackType

    from siso.types import FloatArray, f32d
    from siso.util.field_data import FloatFieldData, IntFieldData


class FieldDimensionality(Enum):
    Planar = auto()
    Volumetric = auto()
    Unknown = auto()


class NetCdf(api.Source[Basis, Field, Step, DiscreteTopology, Zone[int]]):
    filename: Path
    dataset: Dataset

    volumetric: bool
    valid_domains: tuple[FieldDimensionality, ...]
    periodic: bool
    geodetic: bool
    staggering: api.Staggering

    longitude_name: ClassVar[str]
    latitude_name: ClassVar[str]
    height_name: ClassVar[str]

    def __init__(self, path: Path):
        self.filename = path

    def __enter__(self) -> Self:
        self.dataset = Dataset(self.filename, "r").__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.dataset.__exit__(exc_type, exc_val, exc_tb)

    @property
    def num_timesteps(self) -> int:
        return len(self.dataset.dimensions["Time"])

    @property
    def num_latitude(self) -> int:
        return len(
            self.dataset.dimensions[
                "south_north" if self.staggering == api.Staggering.Inner else "south_north_stag"
            ]
        )

    @property
    def num_longitude(self) -> int:
        return len(
            self.dataset.dimensions[
                "west_east" if self.staggering == api.Staggering.Inner else "west_east_stag"
            ]
        )

    @property
    def num_vertical(self) -> int:
        return len(
            self.dataset.dimensions[
                "bottom_top" if self.staggering == api.Staggering.Inner else "bottom_top_stag"
            ]
        )

    @property
    def num_planar(self) -> int:
        return self.num_latitude * self.num_longitude

    @property
    def wrf_planar_nodeshape(self) -> NodeShape:
        return NodeShape(self.num_latitude, self.num_longitude)

    @property
    def wrf_planar_cellshape(self) -> CellShape:
        return self.wrf_planar_nodeshape.cellular

    @property
    def wrf_nodeshape(self) -> NodeShape:
        planar_shape = self.wrf_planar_nodeshape
        if not self.volumetric:
            return planar_shape
        return NodeShape(self.num_vertical, *planar_shape)

    @property
    def wrf_cellshape(self) -> CellShape:
        return self.wrf_nodeshape.cellular

    def field_domain(self, name: str) -> FieldDimensionality:
        try:
            time, *dimensions = self.dataset[name].dimensions
        except IndexError:
            return FieldDimensionality.Unknown

        if time != "Time":
            return FieldDimensionality.Unknown

        try:
            x, y = dimensions
            assert x.startswith("south_north")
            assert y.startswith("west_east")
            return FieldDimensionality.Planar
        except (AssertionError, ValueError):
            pass

        try:
            x, y, z = dimensions
            assert x.startswith("bottom_top")
            assert y.startswith("south_north")
            assert z.startswith("west_east")
            return FieldDimensionality.Volumetric
        except (AssertionError, ValueError):
            pass

        return FieldDimensionality.Unknown

    def field_data_raw(
        self,
        name: str,
        index: int,
        extrude_if_volumetric: bool = True,
        include_poles_if_periodic: bool = True,
    ) -> FieldData[f32d]:
        time, *dimensions = self.dataset[name].dimensions
        assert time == "Time"
        assert len(dimensions) in (2, 3)
        dimensions = list(dimensions)
        data = self.dataset[name][index, ...].filled()

        # Handle staggering
        for dim, dim_name in enumerate(dimensions):
            if dim_name.endswith("_stag") and self.staggering == api.Staggering.Inner:
                data = util.unstagger(data, dim)
                dimensions[dim] = dim_name[:-5]
            elif not dim_name.endswith("_stag") and self.staggering == api.Staggering.Outer:
                data = util.stagger(data, dim)
                dimensions[dim] = f"{dim}_stag"

        if len(dimensions) == 3 and not self.volumetric:
            index = len(self.dataset.dimensions["soil_layers_stag"]) - 1
            data = data[index, ...]
            dimensions = dimensions[1:]

        south: np.ndarray | None = None
        north: np.ndarray | None = None
        if include_poles_if_periodic and self.periodic:
            if name in (self.longitude_name, self.latitude_name):
                south = util.angular_mean(data[0])
                north = util.angular_mean(data[-1])
            else:
                south = np.mean(data[..., 0, :], axis=-1)
                north = np.mean(data[..., -1, :], axis=-1)

        data = data.reshape(self.num_vertical, -1) if len(dimensions) == 3 else data.flatten()

        if include_poles_if_periodic and self.periodic:
            to_append = np.array([south, north]).T
            data = np.append(data, to_append, axis=-1)

        if len(dimensions) == 2 and extrude_if_volumetric and self.volumetric:
            newdata = np.zeros_like(data, shape=(self.num_vertical,) + data.shape)
            newdata[...] = data
            data = newdata

        return FieldData(data.reshape(-1, 1))

    def periodic_planar_topology(self) -> IntFieldData:
        cells = [util.structured_cells(self.wrf_planar_cellshape, pardim=2)]

        nodemap = util.nodemap((self.num_latitude, 2), (self.num_longitude, self.num_longitude - 1))
        cells.append(util.structured_cells(CellShape(self.num_latitude - 1, 1), pardim=2, nodemap=nodemap))

        south_pole_id = self.num_planar
        nodemap = util.nodemap((2, self.num_longitude + 1), (south_pole_id, 1), periodic=(1,))
        nodemap[1] = nodemap[1, 0]
        cells.append(util.structured_cells(CellShape(1, self.num_longitude), pardim=2, nodemap=nodemap))

        north_pole_id = self.num_planar + 1
        nodemap = util.nodemap(
            (2, self.num_longitude + 1), (-self.num_longitude - 1, 1), periodic=(1,), init=north_pole_id
        )
        nodemap[0] = nodemap[0, 0]
        cells.append(util.structured_cells(CellShape(1, self.num_longitude), pardim=2, nodemap=nodemap))

        return FieldData.join_dofs(cells)

    def periodic_volumetric_topology(self) -> IntFieldData:
        cells = [util.structured_cells(self.wrf_cellshape, pardim=3)]

        cells[0] += cells[0] // self.num_planar * 2
        num_horizontal = self.num_planar + 2

        nodemap = util.nodemap(
            (self.num_vertical, self.num_latitude, 2),
            (num_horizontal, self.num_longitude, self.num_longitude - 1),
        )
        cells.append(
            util.structured_cells(
                CellShape(self.num_vertical - 1, self.num_latitude - 1, 1), pardim=3, nodemap=nodemap
            )
        )

        south_pole_id = self.num_planar
        nodemap = util.nodemap(
            (self.num_vertical, 2, self.num_longitude + 1), (num_horizontal, south_pole_id, 1), periodic=(2,)
        )
        nodemap[:, 1] = (nodemap[:, 1] - south_pole_id) // num_horizontal * num_horizontal + south_pole_id
        cells.append(
            util.structured_cells(
                CellShape(self.num_vertical - 1, 1, self.num_longitude), pardim=3, nodemap=nodemap
            )
        )

        north_pole_id = self.num_planar + 1
        nodemap = util.nodemap(
            (self.num_vertical, 2, self.num_longitude + 1),
            (num_horizontal, -self.num_longitude - 1, 1),
            periodic=(2,),
            init=north_pole_id,
        )
        nodemap[:, 0] = (nodemap[:, 0] - north_pole_id) // num_horizontal * num_horizontal + north_pole_id
        cells.append(
            util.structured_cells(
                CellShape(self.num_vertical - 1, 1, self.num_longitude), pardim=3, nodemap=nodemap
            )
        )

        return FieldData.join_dofs(cells)

    def rotation(self, with_intrinsic: bool = True) -> Rotation:
        intrinsic = 0.0
        if with_intrinsic:
            intrinsic = 360 * np.ceil(self.num_longitude / 2) / self.num_longitude
        return Rotation.from_euler(
            "ZYZ", [-self.dataset.STAND_LON, -self.dataset.MOAD_CEN_LAT, intrinsic], degrees=True
        )

    @impl_use_geometry
    def use_geometry(self, geometry: Field) -> None:
        self.geodetic = geometry.name == "Geodetic"

    def zones(self) -> Iterator[Zone]:
        corners = FieldData.join_comps(
            self.field_data_raw(
                self.longitude_name, 0, extrude_if_volumetric=False, include_poles_if_periodic=False
            ),
            self.field_data_raw(
                self.latitude_name, 0, extrude_if_volumetric=False, include_poles_if_periodic=False
            ),
        ).corners(self.wrf_planar_nodeshape)

        yield Zone(
            shape=ZoneShape.Hexahedron,
            coords=corners,
            key=0,
        )

    def bases(self) -> Iterator[Basis]:
        yield Basis("mesh")

    @impl_basis_of
    def basis_of(self, field: Field) -> Basis:
        return Basis("mesh")

    @impl_geometries
    def geometries(self, basis: Basis) -> Iterator[Field]:
        yield Field("Generic", type=api.Geometry(num_comps=3, coords=Generic()))
        yield Field("Geodetic", type=api.Geometry(num_comps=3, coords=Geodetic(Wgs84())))

    @impl_fields
    def fields(self, basis: Basis) -> Iterator[Field]:
        for variable in self.dataset.variables:
            if self.field_domain(variable) in self.valid_domains:
                yield Field(variable, type=api.Scalar())

    @impl_topology
    def topology(self, timestep: Step, basis: Basis, zone: Zone) -> DiscreteTopology:
        if self.periodic:
            num_nodes = self.num_planar + 2
            if self.volumetric:
                num_nodes *= self.num_vertical
                return UnstructuredTopology(
                    num_nodes,
                    self.periodic_volumetric_topology(),
                    celltype=CellType.Hexahedron,
                    degree=1,
                )
            return UnstructuredTopology(
                num_nodes,
                self.periodic_planar_topology(),
                celltype=CellType.Quadrilateral,
                degree=1,
            )
        celltype = CellType.Hexahedron if self.volumetric else CellType.Quadrilateral
        return StructuredTopology(self.wrf_cellshape, celltype, degree=1)

    @impl_topology_updates
    def topology_updates(self, step: Step, basis: Basis) -> bool:
        return step.index == 0

    @impl_field_data
    def field_data(self, timestep: Step, field: Field, zone: Zone) -> FloatFieldData:
        if not field.is_geometry and not field.is_vector:
            return self.field_data_raw(field.name, timestep.index)
        assert field.is_geometry
        return self.geometry(timestep.index)

    def height(self, index: int) -> FloatFieldData:
        if self.volumetric:
            return (self.field_data_raw("PH", index) + self.field_data_raw("PHB", index)) / 9.81
        return self.field_data_raw(self.height_name, 0)

    @lru_cache(maxsize=1)
    def geometry(self, index: int) -> FloatFieldData:
        if self.geodetic:
            return FieldData.join_comps(
                self.field_data_raw(self.longitude_name, index),
                self.field_data_raw(self.latitude_name, index),
                self.height(index),
            )

        x = np.zeros(self.wrf_nodeshape, dtype=float)
        y = np.zeros(self.wrf_nodeshape, dtype=float)
        x[...] = np.arange(self.num_longitude)[..., np.newaxis, :] * self.dataset.DX
        y[...] = np.arange(self.num_latitude)[..., :, np.newaxis] * self.dataset.DY

        return FieldData.join_comps(
            FieldData(x.reshape(-1, 1)),
            FieldData(y.reshape(-1, 1)),
            self.height(index),
        )


class Wrf(NetCdf):
    longitude_name = "XLONG"
    latitude_name = "XLAT"
    height_name = "HGT"

    @staticmethod
    def applicable(path: Path) -> bool:
        try:
            with Dataset(path, "r") as f:
                assert "WRF" in f.TITLE
            return True
        except (AssertionError, OSError):
            return False

    @property
    def properties(self) -> api.SourceProperties:
        return api.SourceProperties(
            instantaneous=False,
            globally_keyed=True,
            discrete_topology=True,
            single_zoned=True,
        )

    def configure(self, settings: api.ReaderSettings) -> None:
        self.volumetric = settings.dimensionality.out_is_volumetric()

        self.valid_domains = (FieldDimensionality.Volumetric,)
        if settings.dimensionality.in_allows_planar():
            self.valid_domains += (FieldDimensionality.Planar,)

        self.staggering = settings.staggering
        self.periodic = settings.periodic

    def steps(self) -> Iterator[Step]:
        for index in range(self.num_timesteps):
            time = self.dataset["XTIME"][index] * 60
            yield Step(index=index, value=time)

    @impl_fields
    def fields(self, basis: Basis) -> Iterator[Field]:
        yield from super().fields(basis)
        yield Field("WIND", type=api.Vector(3, api.VectorInterpretation.Flow), splittable=False)

    @impl_field_data
    def field_data(self, timestep: Step, field: Field, zone: Zone) -> FloatFieldData:
        if field.name == "WIND":
            return self.wind(timestep.index)
        return super().field_data(timestep, field, zone)

    @lru_cache(maxsize=1)
    def wind(self, index: int) -> FloatFieldData:
        local = FieldData.join_comps(
            self.field_data_raw("U", index, include_poles_if_periodic=False),
            self.field_data_raw("V", index, include_poles_if_periodic=False),
            self.field_data_raw("W", index, include_poles_if_periodic=False),
        )

        if not self.geodetic:
            return local

        lonlat = FieldData.join_comps(
            self.field_data_raw(self.longitude_name, index, include_poles_if_periodic=False),
            self.field_data_raw(self.latitude_name, index, include_poles_if_periodic=False),
        )

        points = (
            lonlat.spherical_to_cartesian()
            .rotate(self.rotation(with_intrinsic=True).inv())
            .cartesian_to_spherical(with_radius=False)
        )

        vectors: FloatArray = local.spherical_to_cartesian_vector_field(points).numpy(
            -1, *self.wrf_planar_nodeshape
        )

        south: FloatArray | None = None
        north: FloatArray | None = None
        if self.periodic:
            south = np.mean(vectors[:, 0, ...], axis=-2)[:, np.newaxis, :]
            north = np.mean(vectors[:, -1, ...], axis=-2)[:, np.newaxis, :]

        vectors = vectors.reshape(-1, self.num_planar, 3)

        if self.periodic:
            assert south is not None
            assert north is not None
            vectors = np.append(vectors, south, axis=1)
            vectors = np.append(vectors, north, axis=1)

        fd_vectors: FloatFieldData = FieldData(vectors.reshape(-1, 3))

        lonlat = FieldData.join_comps(
            self.field_data_raw(self.longitude_name, index),
            self.field_data_raw(self.latitude_name, index),
        )

        return fd_vectors.rotate(self.rotation()).cartesian_to_spherical_vector_field(lonlat)


class GeoGrid(NetCdf):
    longitude_name = "XLONG_M"
    latitude_name = "XLAT_M"
    height_name = "HGT_M"

    @staticmethod
    def applicable(path: Path) -> bool:
        try:
            with Dataset(path, "r") as f:
                assert "GEOGRID" in f.TITLE
            return True
        except (AssertionError, OSError):
            return False

    @property
    def properties(self) -> api.SourceProperties:
        return api.SourceProperties(
            instantaneous=True,
            globally_keyed=True,
            discrete_topology=True,
            single_zoned=True,
        )

    def configure(self, settings: api.ReaderSettings) -> None:
        self.volumetric = False
        self.valid_domains = (FieldDimensionality.Planar,)
        self.staggering = settings.staggering
        self.periodic = settings.periodic

    def steps(self) -> Iterator[Step]:
        yield Step(index=0, value=0.0)
