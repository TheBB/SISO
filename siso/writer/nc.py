from netCDF4 import Dataset

import treelog as log

from .writer import TesselatedWriter
from ..fields import Field
from ..geometry import UnstructuredPatch

from ..typing import Array2D


class NetCDFWriter(TesselatedWriter):

    writer_name = "NetCDF4-UGRID"

    @classmethod
    def applicable(cls, fmt: str) -> bool:
        return fmt == 'nc'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.filename = filename
        self.initialized_geometry = False
        # self.finalized_geometry = False
        self.patches = {}

    def __enter__(self):
        self.out = Dataset(self.outpath, 'w').__enter__()
        self.timedim = self.out.createDimension('time')
        self.timevar = self.out.createVariable('time', 'f8', ('time',))
        self.out.Conventions = 'ACDD-1.3, CF-1.7, UGRID-1.0'
        return super().__enter__()

    def __exit__(self, *args):
        self.out.__exit__(*args)
        return super().__exit__(*args)

    def _update_geometry(self, patchid: int, patch: UnstructuredPatch, data: Array2D):
        if self.stepid != 0:
            raise ValueError("NetCDF4 writer currently only supports non-variable geometries")
        if not self.initialized_geometry:
            self.initialize_geometry(data.shape[-1])
        self.patches[patchid] = (data, patch.cells)

    def _update_field(self, field: Field, patchid: int, data: Array2D):
        if field.cells:
            log.warning(f"NetCDF writer doesn't support cell fields: skipping {field.name}")
            return
        if field.ncomps > 1:
            for i, subfield in enumerate(field.decompositions()):
                self._insert_field(subfield, patchid, data[..., i:i+1])
        else:
            self._insert_field(field, patchid, data)

    def _insert_field(self, field, patchid, data):
        try:
            var = self.out[field.name]
        except IndexError:
            var = self.out.createVariable(field.name, data.dtype, ('time', 'Mesh_node'))
        var.mesh = 'Mesh'
        var.location = 'node'

        start = self.patch_starts[patchid]
        var[self.stepid, start:len(data)] = data.flatten()

    def initialize_geometry(self, dim):
        mesh = self.out.createVariable('Mesh', 'i4')
        mesh.cf_role = 'mesh_topology'
        mesh.long_name = f"Topology data of a {dim}D unstructured mesh"
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

    def finalize_geometry(self):
        super().finalize_geometry()

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

    # def update_field(self, results, name, stepid, patchid, kind='scalar', cells=False):
    #     if cells:
    #         return
    #     if kind == 'vector':
    #         for i, n in zip(range(results.shape[-1]), 'xyz'):
    #             self._update_field(results[..., i], '{}_{}'.format(name, n), stepid, patchid)
    #     else:
    #         self._update_field(results, name, stepid, patchid)

    # def finalize_step(self):
    #     pass
