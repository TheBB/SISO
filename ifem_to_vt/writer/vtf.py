from collections import defaultdict, OrderedDict
from pathlib import Path

from singledispatchmethod import singledispatchmethod
import treelog as log

from typing import Optional

from .. import config
from ..fields import AbstractFieldPatch, CombinedFieldPatch
from ..geometry import Patch, UnstructuredPatch
from .writer import Writer

try:
    import vtfwriter as vtf
    HAS_VTF = True
except ImportError:
    HAS_VTF = False


class VTFWriter(Writer):

    writer_name = "VTF"

    out: 'vtf.File'             # vtf is optional
    dirty_geometry: bool

    @classmethod
    def applicable(cls, fmt: str) -> bool:
        return HAS_VTF and fmt == 'vtf'

    def __init__(self, filename: Path):
        super().__init__(filename)
        self.steps = []
        self.geometry_blocks = []
        self.internal_stepid = dict()

        self.mode_data = []
        self.field_blocks = OrderedDict()
        self.dirty_geometry = False

    def validate_mode(self):
        if config.output_mode not in ('ascii', 'binary'):
            raise ValueError("VTF format does not support '{}' mode".format(config.output_mode))

    def __enter__(self) -> 'Writer':
        super().__enter__()
        self.out = vtf.File(str(self.make_filename()), 'w' if config.output_mode == 'ascii' else 'wb').__enter__()
        self.gblock = self.out.GeometryBlock().__enter__()
        return self

    def __exit__(self, *args, **kwargs):
        for fname, data in self.field_blocks.items():
            with data['cons']() as fblock:
                fblock.SetName(fname)
                for stepid, rblocks in data['steps'].items():
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

        if len(self.geometry_blocks) <= patchid:
            self.geometry_blocks.extend([None] * (patchid - len(self.geometry_blocks) + 1))

        with self.out.NodeBlock() as nblock:
            nblock.SetNodes(patch.nodes.flat)

        with self.out.ElementBlock() as eblock:
            eblock.AddElements(patch.cells.flat, patch.num_pardim)
            eblock.SetPartName('Patch {}'.format(patchid+1))
            eblock.BindNodeBlock(nblock, patchid+1)

        self.geometry_blocks[patchid] = (nblock, eblock)
        self.dirty_geometry = True

    def finalize_geometry(self):
        if self.dirty_geometry:
            self.gblock.BindElementBlocks(*[e for _, e in self.geometry_blocks], step=self.stepid+1)
        self.dirty_geometry = False
        super().finalize_geometry()

    def update_field(self, field: AbstractFieldPatch):
        patchid = super().update_field(field)
        field.ensure_ncomps(3, allow_scalar=True)
        if isinstance(field, CombinedFieldPatch) or not isinstance(field.patch, UnstructuredPatch):
            data = field.tesselate()
        else:
            data = field.data

        nblock, eblock = self.geometry_blocks[patchid]
        with self.out.ResultBlock(cells=field.cells, vector=field.num_comps>1) as rblock:
            rblock.SetResults(data.flat)
            rblock.BindBlock(eblock if field.cells else nblock)

        if field.name not in self.field_blocks:
            if field.is_scalar:
                cons = self.out.ScalarBlock
            elif not field.is_displacement:
                cons = self.out.VectorBlock
            else:
                cons = self.out.DisplacementBlock
            self.field_blocks[field.name] = {'cons': cons, 'steps': {}}

        steps = self.field_blocks[field.name]['steps']
        steps.setdefault(self.stepid + 1, []).append(rblock)
