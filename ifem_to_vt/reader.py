import h5py
from collections import namedtuple, OrderedDict
from io import StringIO
from itertools import chain, product
import logging
import numpy as np
import splipy.io
from splipy import SplineObject, BSplineBasis
from splipy.SplineModel import ObjectCatalogue
import splipy.utils


class G2Object(splipy.io.G2):

    def __init__(self, fstream, mode):
        self.fstream = fstream
        self.onlywrite = mode == 'w'
        super(G2Object, self).__init__('')

    def __enter__(self):
        return self

POINTVALUES = 0
CELLVALUES = 1
EIGENMODE = 2

Field = namedtuple('Field', ['name', 'basis', 'ncomps', 'kind'])


class Basis:

    def __init__(self, name, reader):
        self.name = name
        self.reader = reader
        self.update_steps = set()
        self.npatches = 0
        self.patch_cache = {}

    def add_update(self, stepid):
        self.update_steps.add(stepid)

    def update_at(self, stepid):
        return stepid in self.update_steps

    def patch_at(self, stepid, patchid):
        while stepid not in self.update_steps:
            stepid -= 1
        key = (stepid, patchid)
        if key not in self.patch_cache:
            g2data = StringIO(self.reader.h5[str(stepid)][self.name]['basis'][str(patchid+1)][:].tobytes().decode())
            with G2Object(g2data, 'r') as g:
                patch = g.read()[0]
                patch.set_dimension(3)
                self.patch_cache[key] = patch
        return self.patch_cache[key]


class Field:

    def __init__(self, name, basis, reader, cells=False, vectorize=False):
        self.name = name
        self.basis = basis
        self.reader = reader
        self.cells = cells
        self.vectorize = vectorize
        self.decompose = True
        self.update_steps = set()

    def add_update(self, stepid):
        self.update_steps.add(stepid)

    def update(self, w, stepid):
        for patchid in range(self.basis.npatches):
            patch = self.basis.patch_at(stepid, patchid)
            tess, globpatchid = self.reader.geometry.tesselation(patch)
            coeffs = self.coeffs(stepid, patchid)

            results = self.tesselate(patch, tess, coeffs)
            if hasattr(self, 'kind'):
                kind = self.kind
            else:
                kind = 'vector' if results.shape[-1] > 1 else 'scalar'
            w.update_field(results, self.name, stepid, globpatchid, kind, cells=self.cells)

            if self.decompose and results.shape[-1] > 1:
                for i in range(results.shape[-1]):
                    w.update_field(
                        results[...,i], '{}[{}]'.format(self.name, i+1),
                        stepid, globpatchid, 'scalar', cells=self.cells,
                    )

    def coeffs(self, stepid, patchid):
        sub = 'knotspan' if self.cells else 'fields'
        return self.reader.h5[str(stepid)][self.basis.name][sub][self.name][str(patchid+1)][:]

    def tesselate(self, patch, tess, coeffs):
        if self.cells:
            # Make a piecewise constant patch
            bases = [BSplineBasis(1, kts) for kts in patch.knots()]
            shape = tuple(b.num_functions() for b in bases)
            coeffs = splipy.utils.reshape(coeffs, shape, order='F')
            patch = SplineObject(bases, coeffs, False, raw=True)
            tess = [[(a+b)/2 for a, b in zip(t[:-1], t[1:])] for t in tess]
        else:
            coeffs = splipy.utils.reshape(coeffs, patch.shape, order='F')
            patch = SplineObject(patch.bases, coeffs, patch.rational, raw=True)

        if patch.dimension == 1 and self.vectorize:
            patch.set_dimension(3)
            patch.controlpoints[...,-1] = patch.controlpoints[...,0].copy()
            patch.controlpoints[...,0] = 0.0
        elif patch.dimension > 1:
            patch.set_dimension(3)

        return patch(*tess)


class GeometryManager:

    def __init__(self, basis, reader):
        self.basis = basis
        self.reader = reader
        self.tesselations = {}

        logging.debug('Using {} for geometry'.format(basis.name))

    def tesselation(self, patch):
        corners = tuple(tuple(patch.controlpoints[idx]) for idx in product((0,-1), repeat=patch.pardim))
        knots = tuple(tuple(p) for p in patch.knots())
        key = corners + knots

        if key not in self.tesselations:
            self.tesselations[key] = (patch.knots(), len(self.tesselations))
        return self.tesselations[key]

    def update(self, w, stepid):
        if not self.basis.update_at(stepid):
            return

        logging.debug('Updating geometry')

        # FIXME: Here, all patches are updated whenever any of them are updated.
        # Maybe overkill.
        for patchid in range(self.basis.npatches):
            patch = self.basis.patch_at(stepid, patchid)
            tess, globpatchid = self.tesselation(patch)
            nodes = patch(*tess)

            # Elements
            ranges = [range(k-1) for k in nodes.shape[:-1]]
            nidxs = [np.array(q) for q in zip(*product(*ranges))]
            eidxs = np.zeros((len(nidxs[0]), 2**len(nidxs)))
            if len(nidxs) == 1:
                eidxs[:,0] = nidxs[0]
                eidxs[:,1] = nidxs[0] + 1
            elif len(nidxs) == 2:
                i, j = nidxs
                eidxs[:,0] = np.ravel_multi_index((i, j), nodes.shape[:-1])
                eidxs[:,1] = np.ravel_multi_index((i+1, j), nodes.shape[:-1])
                eidxs[:,2] = np.ravel_multi_index((i+1, j+1), nodes.shape[:-1])
                eidxs[:,3] = np.ravel_multi_index((i, j+1), nodes.shape[:-1])
            elif len(nidxs) == 3:
                i, j, k = nidxs
                eidxs[:,0] = np.ravel_multi_index((i, j, k), nodes.shape[:-1])
                eidxs[:,1] = np.ravel_multi_index((i+1, j, k), nodes.shape[:-1])
                eidxs[:,2] = np.ravel_multi_index((i+1, j+1, k), nodes.shape[:-1])
                eidxs[:,3] = np.ravel_multi_index((i, j+1, k), nodes.shape[:-1])
                eidxs[:,4] = np.ravel_multi_index((i, j, k+1), nodes.shape[:-1])
                eidxs[:,5] = np.ravel_multi_index((i+1, j, k+1), nodes.shape[:-1])
                eidxs[:,6] = np.ravel_multi_index((i+1, j+1, k+1), nodes.shape[:-1])
                eidxs[:,7] = np.ravel_multi_index((i, j+1, k+1), nodes.shape[:-1])

            logging.debug('Writing patch {}'.format(globpatchid))
            w.update_geometry(nodes, eidxs, len(nidxs), globpatchid)

        w.finalize_geometry(stepid)


class Reader:

    def __init__(self, h5):
        self.h5 = h5

    def __enter__(self):
        self.bases = OrderedDict()
        self.fields = OrderedDict()
        self.init_bases()
        self.init_fields()
        self.geometry = GeometryManager(next(iter(self.bases.values())), self)
        return self

    def __exit__(self, type_, value, backtrace):
        self.h5.close()

    @property
    def nsteps(self):
        return len(self.h5)

    def steps(self):
        for stepid in range(self.nsteps):
            # FIXME: Grab actual time here as second element
            yield stepid, {'time': float(stepid)}, self.h5[str(stepid)]

    def outputsteps(self):
        for stepid, time, _ in self.steps():
            yield stepid, time

    def init_bases(self):
        for stepid, _, stepgrp in self.steps():
            for basisname in stepgrp:
                if basisname not in self.bases:
                    self.bases[basisname] = Basis(basisname, self)
                basis = self.bases[basisname]

                basis.add_update(stepid)
                basis.npatches = max(basis.npatches, len(stepgrp[basisname]['basis']))

        for basis in self.bases.values():
            logging.debug('Basis {} updates at steps {} ({} patches)'.format(
                basis.name, ', '.join(str(s) for s in sorted(basis.update_steps)), basis.npatches,
            ))

    def init_fields(self):
        for stepid, _, stepgrp in self.steps():
            for basisname, basisgrp in stepgrp.items():
                fields = basisgrp['fields'] if 'fields' in basisgrp else []
                kspans = basisgrp['knotspan'] if 'knotspan' in basisgrp else []

                for fieldname in fields:
                    if fieldname not in self.fields:
                        self.fields[fieldname] = Field(fieldname, self.bases[basisname], self)
                    self.fields[fieldname].add_update(stepid)
                for fieldname in kspans:
                    if fieldname not in self.fields:
                        self.fields[fieldname] = Field(fieldname, self.bases[basisname], self, cells=True)
                    self.fields[fieldname].add_update(stepid)

        for field in self.fields.values():
            logging.debug('{} "{}" lives on {}'.format(
                'Knotspan' if field.cells else 'Field', field.name, field.basis.name
            ))

    def write(self, w):
        for stepid, time in self.outputsteps():
            logging.info('Step {}'.format(stepid))

            w.add_step(**time)
            self.geometry.update(w, stepid)

            for field in self.fields.values():
                logging.debug('Updating {}'.format(field.name))
                field.update(w, stepid)

    def write_mode(self, w, mid, field):
        for pid in range(self.npatches(0, field.basis)):
            patch, tesselation = self._tesselated_patch(0, field.basis, pid)
            coeffs, data = self.mode_coeffs(field, mid, pid)
            raw = self._tesselate(patch, tesselation, coeffs, vectorize=True)
            results = np.ndarray.flatten(raw)
            w.update_mode(results, field.name, pid, **data)

    def modeids(self, basis):
        yield from range(len(self.h5['0'][basis]['Eigenmode']))

    def basis_level(self, level, basis):
        if not isinstance(basis, Basis):
            basis = self.bases[basis]
        try:
            return next(l for l in basis.updates[::-1] if l <= level)
        except StopIteration:
            raise ValueError('Geometry for basis {} unavailable at timestep {}'.format(basis, index))

    def npatches(self, level, basis):
        if not isinstance(basis, Basis):
            basis = self.bases[basis]
        level = self.basis_level(level, basis)
        return len(self.h5['{}/{}/basis'.format(str(level), basis.name)])

    def mode_coeffs(self, field, mid, pid):
        mgrp = self.h5['0'][field.basis.name]['Eigenmode'][str(mid+1)]
        coeffs = mgrp[str(pid+1)][:]
        if 'Value' in mgrp:
            return coeffs, {'value': mgrp['Value'][0]}
        return coeffs, {'frequency': mgrp['Frequency'][0]}


class EigenField(Field):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kind = 'displacement'
        self.decompose = False

    def coeffs(self, stepid, patchid):
        return self.reader.h5['0'][self.basis.name]['Eigenmode'][str(stepid+1)][str(patchid+1)][:]


class EigenReader(Reader):

    @property
    def basis(self):
        return next(iter(self.bases.values()))

    @property
    def nmodes(self):
        return len(self.h5['0'][self.basis.name]['Eigenmode'])

    def outputsteps(self):
        for stepid in range(self.nmodes):
            # FIXME: Grab actual data here as second element
            grp = self.h5['0'][self.basis.name]['Eigenmode'][str(stepid+1)]
            if 'Value' in grp:
                data = {'value': grp['Value'][0]}
            else:
                data = {'frequency': grp['Frequency'][0]}
            yield stepid, data

    def init_fields(self):
        basis = next(iter(self.bases.values()))
        self.fields['Mode Shape'] = EigenField('Mode Shape', basis, self, vectorize=True)


def get_reader(filename):
    h5 = h5py.File(filename, 'r')
    basisname = next(iter(h5['0']))
    if 'Eigenmode' in h5['0'][basisname]:
        logging.info('Detected eigenmodes')
        return EigenReader(h5)
    return Reader(h5)
