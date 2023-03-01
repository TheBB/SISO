from enum import Enum, auto
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
from netCDF4 import Dataset
from typing_extensions import Self

from .. import api, util
from ..field import Field
from ..timestep import TimeStep
from ..topology import CellType, DiscreteTopology, StructuredTopology
from ..util import FieldData
from ..zone import Shape, Zone


class FieldDimensionality(Enum):
    Planar = auto()
    Volumetric = auto()
    Unknown = auto()


class NetCdf:
    filename: Path
    dataset: Dataset

    volumetric: bool
    valid_domains: Tuple[FieldDimensionality, ...]
    staggering: api.Staggering

    def __init__(self, path: Path):
        self.filename = path

    def __enter__(self) -> Self:
        self.dataset = Dataset(self.filename, "r").__enter__()
        return self

    def __exit__(self, *args) -> None:
        self.dataset.__exit__(*args)

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
    def wrf_planar_nodeshape(self) -> Tuple[int, ...]:
        return (self.num_latitude, self.num_longitude)

    @property
    def wrf_nodeshape(self) -> Tuple[int, ...]:
        planar_shape = self.wrf_planar_nodeshape
        if not self.volumetric:
            return planar_shape
        return (self.num_vertical, *planar_shape)

    @property
    def wrf_cellshape(self) -> Tuple[int, ...]:
        return tuple(s - 1 for s in self.wrf_nodeshape)

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
        extrude_if_planar: bool = False,
    ) -> FieldData:
        time, *dimensions = self.dataset[name].dimensions
        assert time == "Time"
        assert len(dimensions) in (2, 3)
        dimensions = list(dimensions)
        data = self.dataset[name][index, ...]

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

        if len(dimensions) == 2 and extrude_if_planar and self.volumetric:
            newdata = np.zeros_like(data, shape=(self.num_vertical,) + data.shape)
            newdata[...] = data
            data = newdata

        return FieldData(data.reshape(-1, 1))


class Wrf(NetCdf):
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
            recombine_fields=[
                api.RecombineFieldSpec(
                    source_names=["U", "V", "W"],
                    new_name="WIND",
                )
            ],
        )

    def configure(self, settings: api.ReaderSettings) -> None:
        self.volumetric = settings.dimensionality.out_is_volumetric()

        self.valid_domains = (FieldDimensionality.Volumetric,)
        if settings.dimensionality.in_allows_planar():
            self.valid_domains += (FieldDimensionality.Planar,)

        self.staggering = settings.staggering

    def use_geometry(self, geometry: Field) -> None:
        return

    def timesteps(self) -> Iterator[TimeStep]:
        for index in range(self.num_timesteps):
            time = self.dataset["XTIME"][index] * 60
            yield TimeStep(index=index, time=time)

    def zones(self) -> Iterator[Zone]:
        corners = FieldData.concat(
            (
                self.field_data_raw("XLONG", 0),
                self.field_data_raw("XLAT", 0),
            )
        ).corners(self.wrf_planar_nodeshape)

        yield Zone(
            shape=Shape.Hexahedron,
            coords=corners,
            local_key="0",
        )

    def fields(self) -> Iterator[Field]:
        yield Field("local", type=api.Geometry(ncomps=3))

        for variable in self.dataset.variables:
            if self.field_domain(variable) in self.valid_domains:
                yield Field(variable, type=api.Scalar(interpretation=api.ScalarInterpretation.Generic))

    def topology(self, timestep: TimeStep, field: Field, zone: Zone) -> DiscreteTopology:
        celltype = CellType.Hexahedron if self.volumetric else CellType.Quadrilateral
        return StructuredTopology(self.wrf_cellshape, celltype)

    def field_data(self, timestep: TimeStep, field: Field, zone: Zone) -> FieldData:
        if not field.is_geometry:
            return self.field_data_raw(field.name, timestep.index, extrude_if_planar=True)

        x = np.zeros(self.wrf_nodeshape, dtype=float)
        y = np.zeros(self.wrf_nodeshape, dtype=float)
        x[...] = np.arange(self.num_longitude)[..., np.newaxis, :] * self.dataset.DX
        y[...] = np.arange(self.num_latitude)[..., :, np.newaxis] * self.dataset.DY

        if self.volumetric:
            height = (
                self.field_data_raw("PH", timestep.index) + self.field_data_raw("PHB", timestep.index)
            ) / 9.81
        else:
            height = self.field_data_raw("HGT", 0)

        return FieldData.concat(
            (
                FieldData(x.reshape(-1, 1)),
                FieldData(y.reshape(-1, 1)),
                height,
            )
        )
