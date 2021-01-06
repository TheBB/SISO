"""Module for VTF format writer."""

from contextlib import contextmanager
from pathlib import Path

from typing import List, Dict, Any, Tuple, Type, Optional

from dataclasses import dataclass
import treelog as log

from .. import config
from ..fields import SimpleField
from ..geometry import Patch
from ..util import ensure_ncomps
from .writer import Writer

from ..typing import Array2D, StepData

try:
    import vtfwriter as vtf
    HAS_VTF = True
except ImportError:
    HAS_VTF = False



@dataclass
class Field:
    """Utility class for block book-keeping."""
    blocktype: Type['vtf.Block']
    steps: Dict[int, List['vtf.ResultBlock']]


class VTFWriter(Writer):
    """Writer for VTF format."""

    writer_name = "VTF"

    out: 'vtf.File'             # vtf is optional
    dirty_geometry: bool

    steps: List[Dict[str, Any]]
    geometry_blocks: List[Tuple['vtf.NodeBlock', 'vtf.ElementBlock']]
    field_blocks: Dict[str, Field]

    gblock: Optional['vtf.GeometryBlock']

    @classmethod
    def applicable(cls, fmt: str) -> bool:
        return HAS_VTF and fmt == 'vtf'

    def __init__(self, filename: Path):
        super().__init__(filename)
        self.steps = []
        self.geometry_blocks = []

        self.field_blocks = dict()
        self.dirty_geometry = False

        self.gblock = None

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
        """Create the state info block, as the last thing to happen before
        closing the file.
        """
        with self.out.StateInfoBlock() as states:
            for stepid, data in enumerate(self.steps):
                key, value = next(iter(data.items()))
                func = states.SetStepData if key == 'time' else states.SetModeData
                desc = {'value': 'Eigenvalue', 'frequency': 'Frequency', 'time': 'Time'}[key]
                func(stepid+1, '{} {:.4g}'.format(desc, value), value)

    @contextmanager
    def step(self, stepdata: StepData):
        self.steps.append(stepdata)
        with super().step(stepdata) as step:
            yield step

    @contextmanager
    def geometry(self, field: Field):
        with super().geometry(field) as geometry:
            yield geometry
            if self.dirty_geometry:
                self.gblock.BindElementBlocks(*[e for _, e in self.geometry_blocks], step=self.stepid+1)
            self.dirty_geometry = False

    def update_geometry(self, geometry: Field, patch: Patch, data: Array2D):
        data = ensure_ncomps(data, 3, allow_scalar=False)
        patchid = patch.key[0]

        # If we haven't seen this patch before, assert that it's the
        # next unseen one
        if len(self.geometry_blocks) <= patchid:
            assert len(self.geometry_blocks) == patchid

        with self.out.NodeBlock() as nblock:
            nblock.SetNodes(data.flat)

        with self.out.ElementBlock() as eblock:
            eblock.AddElements(patch.topology.cells.flat, patch.topology.num_pardim)
            eblock.SetPartName('Patch {}'.format(patchid+1))
            eblock.BindNodeBlock(nblock, patchid+1)

        if len(self.geometry_blocks) <= patchid:
            self.geometry_blocks.append((nblock, eblock))
        else:
            self.geometry_blocks[patchid] = (nblock, eblock)
        self.dirty_geometry = True

    def update_field(self, field: SimpleField, patch: Patch, data: Array2D):
        data = ensure_ncomps(data, 3, allow_scalar=field.is_scalar)
        patchid = patch.key[0]

        nblock, eblock = self.geometry_blocks[patchid]
        with self.out.ResultBlock(cells=field.cells, vector=field.is_vector) as rblock:
            rblock.SetResults(data.flatten())
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
