from netCDF4 import Dataset

import treelog as log

from .writer import Writer
from ..fields import Field
from ..geometry import Patch, Hex, StructuredTopology

from ..typing import Array2D
from typing import Any


class NetCDFCFWriter(Writer):

    writer_name = "NetCDF4-CF"

    @classmethod
    def applicable(cls, fmt: str) -> bool:
        return fmt == 'nc'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialized_geometry = False

    def __enter__(self):
        self.out = Dataset(self.outpath, 'w').__enter__()
        self.timedim = self.out.createDimension('time')
        self.timevar = self.out.createVariable('time', 'f8', ('time',))
        self.timevar.long_name = 'time'
        self.timevar.units = 's'
        self.out.Conventions = 'ACDD-1.3, CF-1.7'
        return super().__enter__()

    def __exit__(self, *args):
        self.out.__exit__(*args)
        log.user(self.outpath)
        return super().__exit__(*args)

    def add_step(self, **stepdata: Any):
        super().add_step(**stepdata)
        if 'time' in stepdata:
            self.out['time'][self.stepid] = stepdata['time']
        else:
            self.out['time'][self.stepid] = self.stepid

    def update_geometry(self, geometry: Field, patch: Patch, data: Array2D):
        if self.initialized_geometry:
            raise TypeError("NetCDF4-CF writer only supports single-patch geometries")
        if self.stepid != 0:
            raise ValueError("NetCDF4 writer currently only supports non-variable geometries")

        topo = patch.topology
        if not isinstance(topo, StructuredTopology):
            raise TypeError("NetCDF4-CF writer only supports structured grids")
        if not isinstance(topo.celltype, Hex):
            raise TypeError("NetCDF4-CF writer only supports hex-cells")

        nodeshape = tuple(s+1 for s in topo.shape)
        self.out.createDimension('i', nodeshape[0])
        self.out.createDimension('j', nodeshape[1])
        self.out.createDimension('k', nodeshape[2])

        x = self.out.createVariable('x', 'f8', ('i', 'j', 'k'))
        x[:] = data[:,0].reshape(nodeshape)
        x.long_name = 'x-coordinate'
        x.units = 'm'

        y = self.out.createVariable('y', 'f8', ('i', 'j', 'k'))
        y[:] = data[:,1].reshape(nodeshape)
        y.long_name = 'y-coordinate'
        y.units = 'm'

        z = self.out.createVariable('z', 'f8', ('i', 'j', 'k'))
        z[:] = data[:,2].reshape(nodeshape)
        z.long_name = 'altitude'
        z.units = 'm'

        self.initialized_geometry = True

    def update_field(self, field: Field, _: Patch, data: Array2D):
        if field.cells:
            log.warning(f"NetCDF writer doesn't support cell fields: skipping {field.name}")
            return
        if field.ncomps > 1:
            for i, subfield in enumerate(field.decompositions()):
                self.insert_field(subfield, data[..., i:i+1])
        else:
            self.insert_field(field, data)

    def insert_field(self, field: Field, data: Array2D):
        try:
            var = self.out[field.name]
        except IndexError:
            var = self.out.createVariable(field.name, data.dtype, ('time', 'i', 'j', 'k'))
        var[self.stepid, ...] = data.reshape(var.shape[1:])

        # TODO: Hardcoded!
        var.long_name = {
            'ps': 'hydrostatic pressure',
            'pt': 'potential temperature',
            'pts': 'potential temperature (hydrostatic)',
            'rho': 'mass density',
            'rhos': 'mass density (hydrostatic)',
            'td': 'energy dissipation rate',
            'tk': 'turbulent kinetic energy',
            'u_x': 'wind speed x-direction',
            'u_y': 'wind speed y-direction',
            'u_z': 'wind speed z-direction',
            'vtef': 'eddy viscosity',
        }[field.name]

        var.units = {
            'ps': 'm2 s-2',
            'pt': 'K',
            'pts': 'K',
            'rho': 'kg m-3',
            'rhos': 'kg m-3',
            'td': 'm2 s-3',
            'tk': 'm2 s-2',
            'u_x': 'm s-1',
            'u_y': 'm s-1',
            'u_z': 'm s-1',
            'vtef': 'm2 s-1',
        }[field.name]


# class NetCDFUgridWriter(TesselatedWriter):

#     writer_name = "NetCDF4-UGRID"

#     @classmethod
#     def applicable(cls, fmt: str) -> bool:
#         return False
#         # return fmt == 'nc'

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.initialized_geometry = False
#         self.patches = {}

#     def __enter__(self):
#         self.out = Dataset(self.outpath, 'w').__enter__()
#         self.timedim = self.out.createDimension('time')
#         self.timevar = self.out.createVariable('time', 'f8', ('time',))
#         self.out.Conventions = 'ACDD-1.3, CF-1.7, UGRID-1.0'
#         return super().__enter__()

#     def __exit__(self, *args):
#         self.out.__exit__(*args)
#         return super().__exit__(*args)

#     def _update_geometry(self, patchid: int, patch: UnstructuredPatch, data: Array2D):
#         if self.stepid != 0:
#             raise ValueError("NetCDF4 writer currently only supports non-variable geometries")
#         if not self.initialized_geometry:
#             self.initialize_geometry(data.shape[-1])
#         self.patches[patchid] = (data, patch.cells)

#     def _update_field(self, field: Field, patchid: int, data: Array2D):
#         if field.cells:
#             log.warning(f"NetCDF writer doesn't support cell fields: skipping {field.name}")
#             return
#         if field.ncomps > 1:
#             for i, subfield in enumerate(field.decompositions()):
#                 self._insert_field(subfield, patchid, data[..., i:i+1])
#         else:
#             self._insert_field(field, patchid, data)

#     def _insert_field(self, field, patchid, data):
#         try:
#             var = self.out[field.name]
#         except IndexError:
#             var = self.out.createVariable(field.name, data.dtype, ('time', 'Mesh_node'))
#         var.mesh = 'Mesh'
#         var.location = 'node'

#         start = self.patch_starts[patchid]
#         var[self.stepid, start:len(data)] = data.flatten()

#     def initialize_geometry(self, dim):
#         mesh = self.out.createVariable('Mesh', 'i4')
#         mesh.cf_role = 'mesh_topology'
#         mesh.long_name = f"Topology data of a {dim}D unstructured mesh"
#         mesh.topology_dimension = 2

#         if dim == 3:
#             mesh.node_coordinates = 'Mesh_x Mesh_y Mesh_z'
#             mesh.volume_node_connectivity = 'Mesh_topology'
#             self.out.createDimension('Mesh_local_nodes', 8)

#             # Only hexahedrons
#             mesh.volume_shape_type = 'Mesh_vol_types'
#             voltypes = self.out.createVariable('Mesh_vol_types', 'i4')
#             voltypes.valid_range = (0, 0)
#             voltypes.flag_values = (0,)
#             voltypes.flag_meanings = 'hexahedron'

#         else:
#             mesh.node_coordinates = 'Mesh_x Mesh_y'
#             mesh.face_node_connectivity = 'Mesh_topology'
#             self.out.createDimension('Mesh_local_nodes', 4)

#     def finalize_geometry(self):
#         super().finalize_geometry()

#         nnodes = sum(nodes.shape[0] for nodes, _ in self.patches.values())
#         nelems = sum(elements.shape[0] for _, elements in self.patches.values())
#         nodedim = self.out.createDimension('Mesh_node', nnodes)
#         elemdim = self.out.createDimension('Mesh_elem', nelems)
#         dim = next(iter(self.patches.values()))[0].shape[-1]

#         coordvars = [self.out.createVariable('Mesh_{}'.format(n), 'f8', ('Mesh_node',)) for n in 'xyz'[:dim]]
#         topovar = self.out.createVariable('Mesh_topology', 'i8', ('Mesh_elem', 'Mesh_local_nodes'))

#         nstart, estart = 0, 0
#         self.patch_starts = {}
#         for patchid, (nodes, elements) in self.patches.items():
#             for coordvar, d in zip(coordvars, range(dim)):
#                 coordvar[nstart:nstart+len(nodes)] = nodes[...,d].reshape((-1,))
#             topovar[estart:estart+len(elements)] = elements
#             self.patch_starts[patchid] = nstart
#             nstart += len(nodes)
#             estart += len(elements)

#         self.patches = {}
