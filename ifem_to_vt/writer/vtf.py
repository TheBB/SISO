from collections import defaultdict, OrderedDict

from .. import config
import vtfwriter as vtf


class Writer(vtf.File):

    def __init__(self, filename):
        if config.output_mode not in ('ascii', 'binary'):
            raise ValueError("VTF format does not support '{}' mode".format(config.output_mode))
        super(Writer, self).__init__(filename, 'w' if config.output_mode == 'ascii' else 'wb')
        self.steps = []
        self.geometry_blocks = []
        self.internal_stepid = dict()

        self.mode_data = []
        self.field_blocks = OrderedDict()
        self.dirty_geometry = False

    def __enter__(self):
        super(Writer, self).__enter__()
        self.gblock = self.GeometryBlock()
        self.gblock.__enter__()
        return self

    def __exit__(self, type_, value, backtrace):
        for fname, data in self.field_blocks.items():
            with data['cons']() as fblock:
                fblock.SetName(fname)
                for stepid, rblocks in data['steps'].items():
                    fblock.BindResultBlocks(stepid, *rblocks)

        self.gblock.__exit__(type_, value, backtrace)
        self.exit_stateinfo()
        super(Writer, self).__exit__(type_, value, backtrace)

    def exit_stateinfo(self):
        with self.StateInfoBlock() as states:
            for stepid, data in enumerate(self.steps):
                key, value = next(iter(data.items()))
                func = states.SetStepData if key == 'time' else states.SetModeData
                desc = {'value': 'Eigenvalue', 'frequency': 'Frequency', 'time': 'Time'}[key]
                func(stepid+1, '{} {:.4f}'.format(desc, value), value)

    def get_stepid(self, stepid):
        return self.internal_stepid.setdefault(stepid, len(self.internal_stepid) + 1)

    def add_step(self, **data):
        self.steps.append(data)

    def finalize_step(self):
        pass

    def update_geometry(self, nodes, elements, dim, patchid):
        if len(self.geometry_blocks) <= patchid:
            self.geometry_blocks.extend([None] * (patchid - len(self.geometry_blocks) + 1))

        with self.NodeBlock() as nblock:
            nblock.SetNodes(nodes.flat)

        with self.ElementBlock() as eblock:
            eblock.AddElements(elements.flat, dim)
            eblock.SetPartName('Patch {}'.format(patchid+1))
            eblock.BindNodeBlock(nblock, patchid+1)

        self.geometry_blocks[patchid] = (nblock, eblock)
        self.dirty_geometry = True

    def finalize_geometry(self, stepid):
        stepid = self.get_stepid(stepid)
        if self.dirty_geometry:
            self.gblock.BindElementBlocks(*[e for _, e in self.geometry_blocks], step=stepid)
        self.dirty_geometry = False

    def update_field(self, results, name, stepid, patchid, kind='scalar', cells=False):
        stepid = self.get_stepid(stepid)
        nblock, eblock = self.geometry_blocks[patchid]
        vector = kind in ('vector', 'displacement')
        with self.ResultBlock(cells=cells, vector=vector) as rblock:
            rblock.SetResults(results.flat)
            rblock.BindBlock(eblock if cells else nblock)

        if name not in self.field_blocks:
            cons = {
                'scalar': self.ScalarBlock,
                'vector': self.VectorBlock,
                'displacement': self.DisplacementBlock,
            }[kind]
            self.field_blocks[name] = {'cons': cons, 'steps': {}}
        steps = self.field_blocks[name]['steps']
        steps.setdefault(stepid, []).append(rblock)
