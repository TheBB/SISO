from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Self

import vtfwriter as vtf
from attrs import define

from siso import api
from siso.api import Basis, CellOrdering, DiscreteTopology, Field, Step, StepInterpretation, Topology, Zone

from .api import OutputMode, Writer, WriterProperties, WriterSettings

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import TracebackType


@define
class FieldInfo:
    blocktype: Callable[[], vtf.ScalarBlock | vtf.VectorBlock | vtf.DisplacementBlock]
    steps: dict[int, list[vtf.ResultBlock]]


class VtfWriter(Writer):
    filename: Path
    out: vtf.File
    mode: OutputMode

    geometry_block: vtf.GeometryBlock
    geometry_blocks: list[tuple[vtf.NodeBlock, vtf.ElementBlock]]
    timesteps: list[Step]
    field_info: dict[str, FieldInfo]
    step_interpretation: StepInterpretation

    def __init__(self, filename: Path):
        self.filename = filename
        self.timesteps = []
        self.geometry_blocks = []
        self.field_info = {}
        self.step_interpretation = StepInterpretation.Time

    @property
    def properties(self) -> WriterProperties:
        return WriterProperties(
            require_discrete_topology=True,
            require_single_basis=True,
        )

    def configure(self, settings: WriterSettings) -> None:
        if settings.output_mode is not None:
            if settings.output_mode not in (OutputMode.Binary, OutputMode.Ascii):
                raise api.Unsupported(f"Unsupported output mode for VTF: {settings.output_mode}")
            self.mode = settings.output_mode
        else:
            self.mode = OutputMode.Binary

    def __enter__(self) -> Self:
        self.out = vtf.File(
            str(self.filename),
            "w" if self.mode == OutputMode.Ascii else "wb",
        ).__enter__()
        self.geometry_block = self.out.GeometryBlock().__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        for field_name, info in self.field_info.items():
            with info.blocktype() as field_block:
                field_block.SetName(field_name)
                for index, result_blocks in info.steps.items():
                    field_block.BindResultBlocks(index, *result_blocks)

        self.geometry_block.__exit__(exc_type, exc_val, exc_tb)

        with self.out.StateInfoBlock() as state_info:
            setter = state_info.SetStepData if self.step_interpretation.is_time else state_info.SetModeData
            desc = str(self.step_interpretation)
            for timestep in self.timesteps:
                time = timestep.value if timestep.value is not None else float(timestep.index)
                setter(timestep.index + 1, f"{desc} {time:.4g}", float(time))

        self.out.__exit__(exc_type, exc_val, exc_tb)
        logging.info(self.filename)

    def update_geometry[B: Basis, F: Field, S: Step](
        self, timestep: S, source: api.Source[B, F, S, DiscreteTopology, Zone[int]], geometry: F
    ) -> None:
        for zone in source.zones():
            topology = source.topology(timestep, source.basis_of(geometry), zone)
            nodes = source.field_data(timestep, geometry, zone).ensure_ncomps(3)

            with self.out.NodeBlock() as node_block:
                node_block.SetNodes(nodes.numpy().flatten())

            with self.out.ElementBlock() as element_block:
                element_block.AddElements(
                    topology.cells_as(CellOrdering.Vtk).numpy().flatten(), topology.pardim
                )
                element_block.SetPartName(f"Patch {zone.key + 1}")
                element_block.BindNodeBlock(node_block, zone.key + 1)

            if len(self.geometry_blocks) <= zone.key:
                self.geometry_blocks.append((node_block, element_block))
            else:
                self.geometry_blocks[zone.key] = (node_block, element_block)

        self.geometry_block.BindElementBlocks(*(e for _, e in self.geometry_blocks), step=timestep.index + 1)

    def update_field[B: Basis, F: Field, S: Step](
        self, timestep: S, source: api.Source[B, F, S, DiscreteTopology, Zone[int]], field: F
    ) -> None:
        for zone in source.zones():
            data = source.field_data(timestep, field, zone)
            data = data.ensure_ncomps(3, allow_scalar=field.is_scalar, pad_right=not field.is_displacement)
            node_block, element_block = self.geometry_blocks[zone.key]

            with self.out.ResultBlock(cells=field.cellwise, vector=field.is_vector) as result_block:
                result_block.SetResults(data.numpy().flatten())
                result_block.BindBlock(element_block if field.cellwise else node_block)

            blocktype: Callable[[], vtf.ScalarBlock | vtf.VectorBlock | vtf.DisplacementBlock]
            if field.name not in self.field_info:
                if field.is_scalar:
                    blocktype = self.out.ScalarBlock
                elif not field.is_displacement:
                    blocktype = self.out.VectorBlock
                else:
                    blocktype = self.out.DisplacementBlock
                self.field_info[field.name] = FieldInfo(blocktype, {})

            steps = self.field_info[field.name].steps
            steps.setdefault(timestep.index + 1, []).append(result_block)

    def consume_timestep[B: Basis, F: Field, S: Step](
        self, timestep: S, source: api.Source[B, F, S, DiscreteTopology, Zone[int]], geometry: F
    ) -> None:
        if source.field_updates(timestep, geometry):
            self.update_geometry(timestep, source, geometry)
        for field in source.fields(source.single_basis()):
            if source.field_updates(timestep, field):
                self.update_field(timestep, source, field)

    def consume[B: Basis, F: Field, S: Step, T: Topology, Z: Zone](
        self, source: api.Source[B, F, S, T, Z], geometry: F
    ) -> None:
        casted = source.cast_discrete_topology().cast_globally_keyed()
        self.step_interpretation = source.properties.step_interpretation
        for step in casted.steps():
            self.timesteps.append(step)
            self.consume_timestep(step, casted, geometry)
