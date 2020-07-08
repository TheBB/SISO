from netCDF4 import Dataset


class Writer:

    def __init__(self, filename, **kwargs):
        self.filename = filename
        self.initialized_geometry = False
        self.finalized_geometry = False
        self.patches = {}

    def __enter__(self):
        self.out = Dataset(self.filename, 'w').__enter__()
        self.timedim = self.out.createDimension('time')
        self.timevar = self.out.createVariable('time', 'f8', ('time',))
        self.out.Conventions = 'ACDD-1.3, CF-1.7, UGRID-1.0'
        return self

    def __exit__(self, type_, value, backtrace):
        self.out.__exit__(type_, value, backtrace)

    def add_step(self, time=None, **kwargs):
        if time is None:
            time = len(self.timevar)
        self.timevar[len(self.timedim)] = time

    def initialize_geometry(self, dim):
        mesh = self.out.createVariable('Mesh', 'i4')
        mesh.cf_role = 'mesh_topology'
        mesh.long_name = 'Topology data of a {}D unstructured mesh'.format(dim)
        mesh.topology_dimension = 2

        if dim == 3:
            mesh.node_coordinates = 'Mesh_x Mesh_y Mesh_z'
            mesh.volume_node_connectivity = 'Mesh_topology'
            self.out.createDimension('Mesh_local_nodes', 8)

            # Only hexahedrons
            mesh.volume_shape_type = 'Mesh_vol_types'
            voltypes = self.out.createVariable('Mesh_vol_types', 'i4')
            voltypes.valid_range = (0, 0)
            voltypes.flag_values = (0,)
            voltypes.flag_meanings = 'hexahedron'

        else:
            mesh.node_coordinates = 'Mesh_x Mesh_y'
            mesh.face_node_connectivity = 'Mesh_topology'
            self.out.createDimension('Mesh_local_nodes', 4)

    def update_geometry(self, nodes, elements, dim, patchid):
        if not self.initialized_geometry:
            self.initialize_geometry(dim)
        self.patches[patchid] = (nodes.reshape((-1, nodes.shape[-1])), elements.reshape((-1, elements.shape[-1])))

    def finalize_geometry(self, stepid):
        if self.finalized_geometry:
            raise ValueError('NetCDF4 writer currently only supports non-variable geometries')

        nnodes = sum(nodes.shape[0] for nodes, _ in self.patches.values())
        nelems = sum(elements.shape[0] for _, elements in self.patches.values())
        nodedim = self.out.createDimension('Mesh_node', nnodes)
        elemdim = self.out.createDimension('Mesh_elem', nelems)
        dim = next(iter(self.patches.values()))[0].shape[-1]

        coordvars = [self.out.createVariable('Mesh_{}'.format(n), 'f8', ('Mesh_node',)) for n in 'xyz'[:dim]]
        topovar = self.out.createVariable('Mesh_topology', 'i8', ('Mesh_elem', 'Mesh_local_nodes'))

        nstart, estart = 0, 0
        self.patch_starts = {}
        for patchid, (nodes, elements) in self.patches.items():
            for coordvar, d in zip(coordvars, range(dim)):
                coordvar[nstart:nstart+len(nodes)] = nodes[...,d].reshape((-1,))
            topovar[estart:estart+len(elements)] = elements
            self.patch_starts[patchid] = nstart
            nstart += len(nodes)
            estart += len(elements)

        self.patches = {}
        self.finalized_geometry = True

    def update_field(self, results, name, stepid, patchid, kind='scalar', cells=False):
        if cells:
            return
        if kind == 'vector':
            for i, n in zip(range(results.shape[-1]), 'xyz'):
                self._update_field(results[..., i], '{}_{}'.format(name, n), stepid, patchid)
        else:
            self._update_field(results, name, stepid, patchid)

    def _update_field(self, results, name, stepid, patchid):
        try:
            var = self.out[name]
        except IndexError:
            var = self.out.createVariable(name, results.dtype, ('time', 'Mesh_node'))
        var.mesh = 'Mesh'
        var.location = 'node'

        results = results.flat
        start = self.patch_starts[patchid]
        var[stepid, start:len(results)] = results

    def finalize_step(self):
        pass
