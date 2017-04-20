import vtfwriter as vtf


class Writer(vtf.File):

    def __init__(self, filename):
        super(Writer, self).__init__(filename, 'w')
        self.times = []
        self.geometry_blocks = []
        self.dirty_geometry = False
        self.patch_counter = 0

    def __enter__(self):
        super(Writer, self).__enter__()
        return self

    def __exit__(self, type_, value, backtrace):
        with self.StateInfoBlock() as states:
            for level, time in enumerate(self.times):
                states.SetStepData(level + 1, 'Time {:.2f}'.format(time), time)
        super(Writer, self).__exit__(type_, value, backtrace)

    def add_time(self, time):
        self.times.append(time)
        if self.dirty_geometry:
            with self.GeometryBlock() as gblock:
                gblock.BindElementBlocks(*[e for _, e in self.geometry_blocks])
            self.dirty_geometry = False

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
