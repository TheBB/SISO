from collections import defaultdict

import vtfwriter as vtf


class Writer(vtf.File):

    def __init__(self, filename):
        super(Writer, self).__init__(filename, 'w')
        self.times = []
        self.mode_data = []
        self.geometry_blocks = []
        self.field_blocks = {}
        self.dirty_geometry = False
        self.patch_counter = 0

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

        if self.mode_data:
            self.clean_geometry(1)
            with self.StateInfoBlock() as states:
                for level, data in enumerate(self.mode_data):
                    if 'value' in data:
                        desc, value = 'Eigenvalue', data['value']
                    else:
                        desc, value = 'Frequency', data['frequency']
                    desc = '{} {:.2e}'.format(desc, value)
                    states.SetModeData(level + 1, desc, value)
        else:
            with self.StateInfoBlock() as states:
                for level, time in enumerate(self.times):
                    states.SetStepData(level + 1, 'Time {:.2f}'.format(time), time)
        self.gblock.__exit__(type_, value, backtrace)
        super(Writer, self).__exit__(type_, value, backtrace)

    def clean_geometry(self, stepid):
        if self.dirty_geometry:
            self.gblock.BindElementBlocks(*[e for _, e in self.geometry_blocks], step=stepid)
        self.dirty_geometry = False

    def add_time(self, time):
        self.times.append(time)
        self.clean_geometry(len(self.times))

    def update_mode(self, results, name, pid, **kwargs):
        self.mode_data.append(kwargs)
        self.update_field(results, name, pid, kind='displacement', stepid=len(self.mode_data))

    def update_field(self, results, name, pid, kind='scalar', cells=False, stepid=None):
        nblock, eblock = self.geometry_blocks[pid]
        vector = kind in ('vector', 'displacement')
        with self.ResultBlock(cells=cells, vector=vector) as rblock:
            rblock.SetResults(results)
            rblock.BindBlock(eblock if cells else nblock)

        if name not in self.field_blocks:
            cons = {
                'scalar': self.ScalarBlock,
                'vector': self.VectorBlock,
                'displacement': self.DisplacementBlock,
            }[kind]
            self.field_blocks[name] = {'cons': cons, 'steps': {}}
        steps = self.field_blocks[name]['steps']
        if stepid is None:
            stepid = len(self.times)
        steps.setdefault(stepid, []).append(rblock)

    def update_geometry(self, nodes, elements, dim, pid=None):
        if pid is None:
            pid = self.patch_counter
            self.patch_counter += 1
            self.geometry_blocks.append(None)
        with self.NodeBlock() as nblock:
            nblock.SetNodes(nodes)
        with self.ElementBlock() as eblock:
            eblock.AddElements(elements, dim)
            eblock.SetPartName('Patch {}'.format(pid+1))
            eblock.BindNodeBlock(nblock)
        self.geometry_blocks[pid] = (nblock, eblock)
        self.dirty_geometry = True
        return pid
