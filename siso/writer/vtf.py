from collections import defaultdict, OrderedDict
from pathlib import Path

from dataclasses import dataclass
import numpy as np
from singledispatchmethod import singledispatchmethod
import treelog as log

from typing import Optional, List, Dict, Any, Tuple, Type
from ..typing import Array2D

from .. import config
from ..fields import Field, SimpleField, CombinedField, PatchData, FieldData
from ..geometry import Patch, UnstructuredPatch
from ..util import ensure_ncomps
from .writer import Writer

try:
    import vtfwriter as vtf
    HAS_VTF = True
except ImportError:
    HAS_VTF = False



@dataclass
class Field:
    blocktype: Type['vtf.Block']
    steps: Dict[int, List['vtf.ResultBlock']]


class VTFWriter(Writer):

    writer_name = "VTF"

    out: 'vtf.File'             # vtf is optional
    dirty_geometry: bool

    steps: List[Dict[str, Any]]
    geometry_blocks: List[Tuple['vtf.NodeBlock', 'vtf.ElementBlock']]
    field_blocks: Dict[str, Field]

    @classmethod
    def applicable(cls, fmt: str) -> bool:
        return HAS_VTF and fmt == 'vtf'

    def __init__(self, filename: Path):
        super().__init__(filename)
        self.steps = []
        self.geometry_blocks = []

        self.field_blocks = dict()
        self.dirty_geometry = False

    def validate(self):
        config.require_in(reason="not supported by VTF", output_mode=('binary', 'ascii'))

    def __enter__(self) -> 'Writer':
        super().__enter__()
        self.out = vtf.File(str(self.make_filename()), 'w' if config.output_mode == 'ascii' else 'wb').__enter__()
        self.gblock = self.out.GeometryBlock().__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        for fname, data in self.field_blocks.items():
            with data.blocktype() as fblock:
                fblock.SetName(fname)
                for stepid, rblocks in data.steps.items():
                    fblock.BindResultBlocks(stepid, *rblocks)

        self.gblock.__exit__(*args, **kwargs)
        self.exit_stateinfo()
        self.out.__exit__(*args, **kwargs)
        super().__exit__(*args, **kwargs)
        log.user(self.make_filename())

    def exit_stateinfo(self):
        with self.out.StateInfoBlock() as states:
            for stepid, data in enumerate(self.steps):
                key, value = next(iter(data.items()))
                func = states.SetStepData if key == 'time' else states.SetModeData
                desc = {'value': 'Eigenvalue', 'frequency': 'Frequency', 'time': 'Time'}[key]
                func(stepid+1, '{} {:.4f}'.format(desc, value), value)

    def add_step(self, **stepdata):
        super().add_step(**stepdata)
        self.steps.append(stepdata)

    @singledispatchmethod
    def update_geometry(self, patch: Patch, patchid: Optional[int] = None):
        if patchid is None:
            patchid = super().update_geometry(patch)
        return self.update_geometry(patch.tesselate(), patchid=patchid)

    @update_geometry.register(UnstructuredPatch)
    def _(self, patch: UnstructuredPatch, patchid: Optional[int] = None):
        if patchid is None:
            patchid = super().update_geometry(patch)
        patch.ensure_ncomps(3, allow_scalar=False)

        # If we haven't seen this patch before, assert that it's the
        # next unseen one
        if len(self.geometry_blocks) <= patchid:
            assert len(self.geometry_blocks) == patchid

        with self.out.NodeBlock() as nblock:
            nblock.SetNodes(patch.nodes.flat)

        with self.out.ElementBlock() as eblock:
            eblock.AddElements(patch.cells.flat, patch.num_pardim)
            eblock.SetPartName('Patch {}'.format(patchid+1))
            eblock.BindNodeBlock(nblock, patchid+1)

        if len(self.geometry_blocks) <= patchid:
            self.geometry_blocks.append((nblock, eblock))
        else:
            self.geometry_blocks[patchid] = (nblock, eblock)
        self.dirty_geometry = True

    def finalize_geometry(self):
        if self.dirty_geometry:
            self.gblock.BindElementBlocks(*[e for _, e in self.geometry_blocks], step=self.stepid+1)
        self.dirty_geometry = False
        super().finalize_geometry()

    @singledispatchmethod
    def update_field(self, field: Field, patch: PatchData, data: FieldData):
        raise NotImplementedError

    @update_field.register(SimpleField)
    def _(self, field: SimpleField, patch: Patch, data: Array2D):
        patchid = self.geometry.global_id(patch)
        data = ensure_ncomps(data, 3, allow_scalar=field.is_scalar)
        data = patch.tesselate_field(data, cells=field.cells)
        print(field.name, field.is_scalar, data.shape)

        nblock, eblock = self.geometry_blocks[patchid]
        with self.out.ResultBlock(cells=field.cells, vector=field.is_vector) as rblock:
            rblock.SetResults(data.flat)
            rblock.BindBlock(eblock if field.cells else nblock)

        if field.name not in self.field_blocks:
            if field.is_scalar:
                blocktype = self.out.ScalarBlock
            elif not field.is_displacement:
                blocktype = self.out.VectorBlock
            else:
                blocktype = self.out.DisplacementBlock
            self.field_blocks[field.name] = Field(blocktype, {})

        steps = self.field_blocks[field.name].steps
        steps.setdefault(self.stepid + 1, []).append(rblock)

    @update_field.register(CombinedField)
    def _(self, field: CombinedField, patch: List[Patch], data: List[Array2D]):
        patchid = self.geometry.global_id(patch[0])
        data = np.hstack([
            p.tesselate_field(d, cells=field.cells)
            for p, d in zip(patch, data)
        ])
        data = ensure_ncomps(data, 3, allow_scalar=field.is_scalar)

        nblock, eblock = self.geometry_blocks[patchid]
        with self.out.ResultBlock(cells=field.cells, vector=field.is_vector) as rblock:
            rblock.SetResults(data.flat)
            rblock.BindBlock(eblock if field.cells else nblock)

        if field.name not in self.field_blocks:
            if field.is_scalar:
                blocktype = self.out.ScalarBlock
            elif not field.is_displacement:
                blocktype = self.out.VectorBlock
            else:
                blocktype = self.out.DisplacementBlock
            self.field_blocks[field.name] = Field(blocktype, {})

        steps = self.field_blocks[field.name].steps
        steps.setdefault(self.stepid + 1, []).append(rblock)
