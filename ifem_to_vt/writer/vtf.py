from collections import defaultdict

import vtfwriter as vtf


class Writer(vtf.File):

    def __init__(self, filename):
        super(Writer, self).__init__(filename, 'w')
        self.times = []
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
        with self.StateInfoBlock() as states:
            for level, time in enumerate(self.times):
                states.SetStepData(level + 1, 'Time {:.2f}'.format(time), time)
        self.gblock.__exit__(type_, value, backtrace)
        for fblock in self.field_blocks.values():
            fblock.__exit__(type_, value, backtrace)
        super(Writer, self).__exit__(type_, value, backtrace)

    def add_time(self, time):
        lid = len(self.times)
        self.times.append(time)
        if self.dirty_geometry:
            self.gblock.BindElementBlocks(*[e for _, e in self.geometry_blocks], step=lid+1)
        self.dirty_geometry = False

    def update_field(self, results, name, pid, kind='scalar', cells=False):
        nblock, eblock = self.geometry_blocks[pid]
        with self.ResultBlock(cells=cells) as rblock:
            rblock.SetResults(results)
            rblock.BindBlock(eblock if cells else nblock)
        if name not in self.field_blocks:
            if kind == 'scalar':
                fblock = self.ScalarBlock()
            elif kind == 'vector':
                fblock = self.VectorBlock()
            else:
                raise ValueError
            self.field_blocks[name] = fblock
            fblock.__enter__()
            fblock.SetName(name)
        else:
            fblock = self.field_blocks[name]
        fblock.BindResultBlocks(len(self.times), rblock)

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
