from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Type, TypeVar

import vtfwriter as vtf
from attrs import define
from typing_extensions import Self

from ..api import Basis, DiscreteTopology, Field, Source, Step, StepInterpretation, Zone
from .api import OutputMode, Writer, WriterProperties, WriterSettings


@define
class FieldInfo:
    blocktype: Type[vtf.Block]
    steps: Dict[int, List[vtf.ResultBlock]]


B = TypeVar("B", bound=Basis)
F = TypeVar("F", bound=Field)
S = TypeVar("S", bound=Step)
Z = TypeVar("Z", bound=Zone)


class VtfWriter(Writer[B, F, S, Z]):
    filename: Path
    out: vtf.File
    mode: OutputMode

    geometry_block: vtf.GeometryBlock
    geometry_blocks: List[Tuple[vtf.NodeBlock, vtf.ElementBlock]]
    timesteps: List[S]
    field_info: Dict[str, FieldInfo]
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
            require_tesselated=True,
        )

    def configure(self, settings: WriterSettings):
        if settings.output_mode is not None:
            assert settings.output_mode in (OutputMode.Ascii, OutputMode.Binary)
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

    def __exit__(self, *args) -> None:
        for field_name, info in self.field_info.items():
            with info.blocktype() as field_block:
                field_block.SetName(field_name)
                for index, result_blocks in info.steps.items():
                    field_block.BindResultBlocks(index, *result_blocks)

        self.geometry_block.__exit__(*args)

        with self.out.StateInfoBlock() as state_info:
            setter = state_info.SetStepData if self.step_interpretation.is_time else state_info.SetModeData
            desc = str(self.step_interpretation)
            for timestep in self.timesteps:
                time = timestep.value if timestep.value is not None else float(timestep.index)
                setter(timestep.index + 1, f"{desc} {time:.4g}", time)

        self.out.__exit__(*args)
        logging.info(self.filename)

    def update_geometry(self, timestep: S, source: Source[B, F, S, Z], geometry: F) -> None:
        for zone in source.zones():
            assert zone.global_key is not None

            topology = source.topology(timestep, source.basis_of(geometry), zone)
            nodes = source.field_data(timestep, geometry, zone).ensure_ncomps(3)
            assert isinstance(topology, DiscreteTopology)

            with self.out.NodeBlock() as node_block:
                node_block.SetNodes(nodes.numpy().flatten())

            with self.out.ElementBlock() as element_block:
                element_block.AddElements(topology.cells.numpy().flatten(), topology.pardim)
                element_block.SetPartName(f"Patch {zone.global_key + 1}")
                element_block.BindNodeBlock(node_block, zone.global_key + 1)

            if len(self.geometry_blocks) <= zone.global_key:
                self.geometry_blocks.append((node_block, element_block))
            else:
                self.geometry_blocks[zone.global_key] = (node_block, element_block)

        self.geometry_block.BindElementBlocks(*(e for _, e in self.geometry_blocks), step=timestep.index + 1)

    def update_field(self, timestep: S, source: Source[B, F, S, Z], field: F) -> None:
        for zone in source.zones():
            assert zone.global_key is not None

            data = source.field_data(timestep, field, zone)
            data = data.ensure_ncomps(3, allow_scalar=field.is_scalar, pad_right=not field.is_displacement)
            node_block, element_block = self.geometry_blocks[zone.global_key]

            with self.out.ResultBlock(cells=field.cellwise, vector=field.is_vector) as result_block:
                result_block.SetResults(data.numpy().flatten())
                result_block.BindBlock(element_block if field.cellwise else node_block)

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

    def consume_timestep(self, timestep: S, source: Source[B, F, S, Z], geometry: F) -> None:
        if source.field_updates(timestep, geometry):
            self.update_geometry(timestep, source, geometry)
        for basis in source.bases():
            for field in source.fields(basis):
                if source.field_updates(timestep, field):
                    self.update_field(timestep, source, field)

    def consume(self, source: Source[B, F, S, Z], geometry: F):
        self.step_interpretation = source.properties.step_interpretation
        for timestep in source.steps():
            self.timesteps.append(timestep)
            self.consume_timestep(timestep, source, geometry)
