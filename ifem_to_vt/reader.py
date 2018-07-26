import h5py
from contextlib import contextmanager
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


class Log:

    _indent = 0

    @classmethod
    def debug(cls, s, *args, **kwargs):
        return logging.debug(' ' * cls._indent + s, *args, **kwargs)

    @classmethod
    def info(cls, s, *args, **kwargs):
        return logging.info(' ' * cls._indent + s, *args, **kwargs)

    @classmethod
    def warning(cls, s, *args, **kwargs):
        return logging.warning(' ' * cls._indent + s, *args, **kwargs)

    @classmethod
    def error(cls, s, *args, **kwargs):
        return logging.error(' ' * cls._indent + s, *args, **kwargs)

    @classmethod
    def critical(cls, s, *args, **kwargs):
        return logging.warn(' ' * cls._indent + s, *args, **kwargs)

    @classmethod
    @contextmanager
    def indent(cls):
        cls._indent += 4
        yield
        cls._indent -= 4


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

            if self.decompose and self.ncomps > 1:
                for i in range(self.ncomps):
                    w.update_field(
                        results[...,i], '{}_{}'.format(self.name, 'xyz'[i]),
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

        self.ncomps = patch.dimension

        if patch.dimension == 1 and self.vectorize:
            patch.set_dimension(3)
            patch.controlpoints[...,-1] = patch.controlpoints[...,0].copy()
            patch.controlpoints[...,0] = 0.0
        elif patch.dimension > 1:
            patch.set_dimension(3)

        return patch(*tess)


class CombinedField(Field):

    def __init__(self, name, sources, reader):
        assert all(not s.vectorize for s in sources)
        cells = set(s.cells for s in sources)
        assert len(cells) == 1

        self.name = name
        self.reader = reader
        self.cells = next(iter(cells))
        self.vectorize = False
        self.decompose = False
        self.ncomps = len(sources)
        self.sources = sources

        self.update_steps = set()
        for s in sources:
            self.update_steps |= s.update_steps

    def update(self, w, stepid):
        # Assume all bases have the same number of patches,
        # so just grab the indices from the first one
        for patchid in range(self.sources[0].basis.npatches):

            results = []
            for src in self.sources:
                patch = src.basis.patch_at(stepid, patchid)
                tess, globpatchid = self.reader.geometry.tesselation(patch)
                coeffs = src.coeffs(stepid, patchid)
                results.append(src.tesselate(patch, tess, coeffs))

            results = np.concatenate(results, axis=-1)

            if hasattr(self, 'kind'):
                kind = self.kind
            else:
                kind = 'vector'
            w.update_field(results, self.name, stepid, globpatchid, kind, cells=self.cells)


class GeometryManager:

    def __init__(self, basis, reader):
        self.basis = basis
        self.reader = reader
        self.tesselations = {}

        Log.info('Using {} for geometry'.format(basis.name))

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

        Log.info('Updating geometry')

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

            Log.debug('Writing patch {}'.format(globpatchid))
            w.update_geometry(nodes, eidxs, len(nidxs), globpatchid)

        w.finalize_geometry(stepid)


class Reader:

    def __init__(self, h5, bases=(), geometry=None):
        self.h5 = h5
        self.only_bases = bases
        self.geometry_basis = geometry

    def __enter__(self):
        self.bases = OrderedDict()
        self.fields = OrderedDict()
        self.init_bases()
        self.init_fields()

        if self.geometry_basis:
            self.geometry = GeometryManager(self.bases[self.geometry_basis], self)
        else:
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

    def allowed_bases(self, group, items=False):
        if not self.only_bases:
            yield from (group.items() if items else group)
        elif items:
            yield from ((b, v) for b, v in group.items() if b in self.only_bases)
        else:
            yield from (b for b in group if b in self.only_bases)

    def init_bases(self):
        for stepid, _, stepgrp in self.steps():
            for basisname in stepgrp:
                if basisname not in self.bases:
                    self.bases[basisname] = Basis(basisname, self)
                basis = self.bases[basisname]

                if 'basis' in stepgrp[basisname]:
                    basis.add_update(stepid)
                    basis.npatches = max(basis.npatches, len(stepgrp[basisname]['basis']))

        # Delete the bases we don't need
        if self.only_bases:
            keep = self.only_bases + ((self.geometry_basis,) if self.geometry_basis else ())
            self.bases = OrderedDict((b,v) for b,v in self.bases.items() if b in keep)

        for basis in self.bases.values():
            Log.debug('Basis {} updates at steps {} ({} patches)'.format(
                basis.name, ', '.join(str(s) for s in sorted(basis.update_steps)), basis.npatches,
            ))

    def init_fields(self):
        for stepid, _, stepgrp in self.steps():
            for basisname, basisgrp in self.allowed_bases(stepgrp, items=True):
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
            Log.debug('{} "{}" lives on {}'.format(
                'Knotspan' if field.cells else 'Field', field.name, field.basis.name
            ))

        # Detect combined fields, e.g. u_x, u_y, u_z -> u
        candidates = OrderedDict()
        for fname in self.fields:
            if len(fname) > 2 and fname[-2] == '_' and fname[-1] in 'xyz' and fname[:-2] not in self.fields:
                candidates.setdefault(fname[:-2], []).append(fname)

        for fname, sourcenames in candidates.items():
            if not (1 < len(sourcenames) < 4):
                continue

            sourcenames = sorted(sourcenames, key=lambda s: s[-1])
            sources = [self.fields[s] for s in sourcenames]
            try:
                self.fields[fname] = CombinedField(fname, sources, self)
                Log.info('Creating combined field {} -> {}'.format(', '.join(sourcenames), fname))
            except AssertionError:
                Log.warning('Unable to combine {} -> {}'.format(', '.join(sourcenames), fname))

        # Reorder fields
        fields = sorted(self.fields.values(), key=lambda f: f.name)
        fields = sorted(fields, key=lambda f: f.cells)
        self.fields = OrderedDict((f.name, f) for f in fields)

    def write(self, w):
        for stepid, time in self.outputsteps():
            Log.info('Step {}'.format(stepid))

            with Log.indent():
                w.add_step(**time)
                self.geometry.update(w, stepid)

                for field in self.fields.values():
                    Log.info('Updating {}'.format(field.name))
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


def get_reader(filename, bases=(), geometry=None):
    h5 = h5py.File(filename, 'r')
    basisname = next(iter(h5['0']))
    if 'Eigenmode' in h5['0'][basisname]:
        Log.info('Detected eigenmodes')
        return EigenReader(h5, bases=bases, geometry=geometry)
    return Reader(h5, bases=bases, geometry=geometry)
