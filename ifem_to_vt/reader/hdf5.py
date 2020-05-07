import h5py
from contextlib import contextmanager
from collections import namedtuple, OrderedDict
from io import StringIO
from itertools import chain, product
import logging
import lrspline as lr
import numpy as np
import splipy.io
from splipy import SplineObject, BSplineBasis
from splipy.SplineModel import ObjectCatalogue
import splipy.utils
import sys


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
        return logging.critical(' ' * cls._indent + s, *args, **kwargs)

    @classmethod
    @contextmanager
    def indent(cls):
        cls._indent += 4
        yield
        cls._indent -= 4


def subdivide_linear(knots, nvis):
    out = []
    for left, right in zip(knots[:-1], knots[1:]):
        out.extend(np.linspace(left, right, num=nvis, endpoint=False))
    out.append(knots[-1])
    return out


def subdivide_face(el, nodes, elements, nvis):
    left, bottom = el.start()
    right, top = el.end()
    xs = subdivide_linear((left, right), nvis)
    ys = subdivide_linear((bottom, top), nvis)

    for (l, r) in zip(xs[:-1], xs[1:]):
        for (b, t) in zip(ys[:-1], ys[1:]):
            sw, se, nw, ne = (l, b), (r, b), (l, t), (r, t)
            for pt in (sw, se, nw, ne):
                nodes.setdefault(pt, len(nodes))
            elements.append([nodes[sw], nodes[se], nodes[ne], nodes[nw]])


def subdivide_volume(el, nodes, elements, nvis):
    umin, vmin, wmin = el.start()
    umax, vmax, wmax = el.end()
    us = subdivide_linear((umin, umax), nvis)
    vs = subdivide_linear((vmin, vmax), nvis)
    ws = subdivide_linear((wmin, wmax), nvis)

    for (ul, ur) in zip(us[:-1], us[1:]):
        for (vl, vr) in zip(vs[:-1], vs[1:]):
            for (wl, wr) in zip(ws[:-1], ws[1:]):
                bsw, bse, bnw, bne = (ul, vl, wl), (ur, vl, wl), (ul, vr, wl), (ur, vr, wl)
                tsw, tse, tnw, tne = (ul, vl, wr), (ur, vl, wr), (ul, vr, wr), (ur, vr, wr)
                for pt in (bsw, bse, bnw, bne, tsw, tse, tnw, tne):
                    nodes.setdefault(pt, len(nodes))
                elements.append([nodes[bsw], nodes[bse], nodes[bne], nodes[bnw],
                                 nodes[tsw], nodes[tse], nodes[tne], nodes[tnw]])


class TensorTesselation:

    def __init__(self, patch, nvis=1):
        self.knots = tuple(subdivide_linear(knots, nvis) for knots in patch.knots())

    def __call__(self, patch, coeffs=None, cells=False):
        if cells:
            assert coeffs is not None
            bases = [BSplineBasis(1, kts) for kts in patch.knots()]
            shape = tuple(b.num_functions() for b in bases)
            coeffs = splipy.utils.reshape(coeffs, shape, order='F')
            patch = SplineObject(bases, coeffs, False, raw=True)
            knots = [[(a+b)/2 for a, b in zip(t[:-1], t[1:])] for t in self.knots]
            return patch(*knots)

        if coeffs is not None:
            coeffs = splipy.utils.reshape(coeffs, patch.shape, order='F')
            if patch.rational:
                coeffs = np.concatenate((coeffs, patch.controlpoints[..., -1, np.newaxis]), axis=-1)
            patch = SplineObject(patch.bases, coeffs, patch.rational, raw=True)
        return patch(*self.knots)

    def elements(self):
        nshape = tuple(len(k) for k in self.knots)
        ranges = [range(k-1) for k in nshape]
        nidxs = [np.array(q) for q in zip(*product(*ranges))]
        eidxs = np.zeros((len(nidxs[0]), 2**len(nidxs)), dtype=int)
        if len(nidxs) == 1:
            eidxs[:,0] = nidxs[0]
            eidxs[:,1] = nidxs[0] + 1
        elif len(nidxs) == 2:
            i, j = nidxs
            eidxs[:,0] = np.ravel_multi_index((i, j), nshape)
            eidxs[:,1] = np.ravel_multi_index((i+1, j), nshape)
            eidxs[:,2] = np.ravel_multi_index((i+1, j+1), nshape)
            eidxs[:,3] = np.ravel_multi_index((i, j+1), nshape)
        elif len(nidxs) == 3:
            i, j, k = nidxs
            eidxs[:,0] = np.ravel_multi_index((i, j, k), nshape)
            eidxs[:,1] = np.ravel_multi_index((i+1, j, k), nshape)
            eidxs[:,2] = np.ravel_multi_index((i+1, j+1, k), nshape)
            eidxs[:,3] = np.ravel_multi_index((i, j+1, k), nshape)
            eidxs[:,4] = np.ravel_multi_index((i, j, k+1), nshape)
            eidxs[:,5] = np.ravel_multi_index((i+1, j, k+1), nshape)
            eidxs[:,6] = np.ravel_multi_index((i+1, j+1, k+1), nshape)
            eidxs[:,7] = np.ravel_multi_index((i, j+1, k+1), nshape)

        return len(nidxs), eidxs


class LRSurfaceTesselation:

    def __init__(self, patch, nvis=1):
        nodes, elements = OrderedDict(), []
        for el in patch.elements:
            subdivide_face(el, nodes, elements, nvis)

        self._nodes = nodes
        self._elements = np.array(elements, dtype=int)
        self.nvis = nvis

    def __call__(self, patch, coeffs=None, cells=False):
        if cells:
            assert coeffs is not None
            ncomps = len(patch.elements) // coeffs.size
            coeffs = coeffs.reshape((-1, ncomps))
            coeffs = np.repeat(coeffs, self.nvis**2, axis=0)
            return coeffs

        if coeffs is not None:
            patch = patch.clone()
            patch.controlpoints = coeffs.reshape((len(patch.basis), -1))

        return np.array([patch(*node) for node in self._nodes], dtype=float)

    def elements(self):
        return 2, self._elements


class LRVolumeTesselation:

    def __init__(self, patch, nvis=1):
        nodes, elements = OrderedDict(), []
        for el in patch.elements:
            subdivide_volume(el, nodes, elements, nvis)

        self._nodes = nodes
        self._elements = np.array(elements, dtype=int)
        self.nvis = nvis

    def __call__(self, patch, coeffs=None, cells=False):
        if cells:
            assert coeffs is not None
            ncomps = len(patch.elements) // coeffs.size
            coeffs = coeffs.reshape((-1, ncomps))
            coeffs = np.repeat(coeffs, self.nvis**3, axis=0)
            return coeffs

        if coeffs is not None:
            patch = patch.clone()
            patch.controlpoints = coeffs.reshape((len(patch.basis), -1))

        return np.array([patch(*node) for node in self._nodes], dtype=float)

    def elements(self):
        return 3, self._elements


def get_tesselation(patch, nvis=1):
    if isinstance(patch, SplineObject):
        return TensorTesselation(patch, nvis=nvis)
    if isinstance(patch, lr.LRSplineSurface):
        return LRSurfaceTesselation(patch, nvis=nvis)
    if isinstance(patch, lr.LRSplineVolume):
        return LRVolumeTesselation(patch, nvis=nvis)
    assert False


def expand_to_dims(array, dims=3):
    nshape = array.shape[:-1] + (1,)
    while array.shape[-1] < dims:
        array = np.concatenate((array, np.zeros(nshape)), axis=-1)
    return array


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
            g2bytes = self.reader.h5[str(stepid)][self.name]['basis'][str(patchid+1)][:].tobytes()
            if g2bytes.startswith(b'# LRSPLINE SURFACE'):
                patch = lr.LRSplineSurface(g2bytes)
            elif g2bytes.startswith(b'# LRSPLINE VOLUME'):
                patch = lr.LRSplineVolume(g2bytes)
            else:
                g2data = StringIO(g2bytes.decode())
                with G2Object(g2data, 'r') as g:
                    patch = g.read()[0]
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
        self.ncomps = None

    @property
    def basisname(self):
        return self.basis.name

    def add_update(self, stepid):
        self.update_steps.add(stepid)
        if self.ncomps is None:
            patch = self.basis.patch_at(stepid, 0)
            coeffs = self.coeffs(stepid, 0)
            if not self.cells:
                num, denom = len(coeffs), len(patch)
            elif isinstance(patch, SplineObject):
                ncells = np.prod([len(k)-1 for k in patch.knots()])
                num, denom = len(coeffs), ncells
            else:
                ncells = len(patch.elements)
                num, denom = len(coeffs), ncells

            if num % denom != 0:
                Log.critical('Inconsistent dimension in field "{}" ({}/{}); unable to discover number of components'.format(
                    self.name, num, denom
                ))
                sys.exit(2)
            self.ncomps = num // denom

    def update_at(self, stepid):
        return stepid in self.update_steps

    def update(self, w, stepid):
        for patchid in range(self.basis.npatches):
            patch = self.basis.patch_at(stepid, patchid)
            tess = self.reader.geometry.tesselation(patch)
            globpatchid = self.reader.geometry.findid(patch)
            if globpatchid is None:
                continue

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
        results = tess(patch, coeffs=coeffs, cells=self.cells)

        if self.ncomps == 1 and self.vectorize:
            results = expand_to_dims(results)
            results[...,-1] = results[...,0].copy()
            results[...,0] = 0.0
        elif self.ncomps > 1:
            results = expand_to_dims(results)

        return results


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

    @property
    def basisname(self):
        return ','.join(source.basis.name for source in self.sources)

    def update(self, w, stepid):
        # Assume all bases have the same number of patches,
        # so just grab the indices from the first one
        for patchid in range(self.sources[0].basis.npatches):

            results = []
            for src in self.sources:
                patch = src.basis.patch_at(stepid, patchid)
                tess = self.reader.geometry.tesselation(patch)
                globpatchid = self.reader.geometry.findid(patch)
                coeffs = src.coeffs(stepid, patchid)
                results.append(src.tesselate(patch, tess, coeffs))

            if globpatchid is None:
                continue

            results = np.concatenate(results, axis=-1)
            results = expand_to_dims(results)

            if hasattr(self, 'kind'):
                kind = self.kind
            else:
                kind = 'vector'

            w.update_field(results, self.name, stepid, globpatchid, kind, cells=self.cells)


class SplitField(Field):

    def __init__(self, name, source, index, reader):
        self.name = name
        self.source = source
        self.basis = source.basis
        self.index = index
        self.reader = reader
        self.cells = source.cells
        self.vectorize = False
        self.decompose = False
        self.ncomps = 1
        self.update_steps = source.update_steps

    def coeffs(self, stepid, patchid):
        coeffs = self.source.coeffs(stepid, patchid)
        coeffs = np.reshape(coeffs, (-1, self.source.ncomps))
        return coeffs[:, self.index]


class GeometryManager:

    def __init__(self, basis, nvis):
        self.basis = basis
        self.nvis = nvis

        # Map knot vector -> evaluation points
        self.tesselations = {}

        # Map basisname x patchid ->
        self.globids = {}

        # Map corner tuple -> basisname x patchid
        self.corners = {}

        Log.info('Using {} for geometry'.format(basis.name))

    def tesselation(self, patch):
        knots = tuple(tuple(k) for k in patch.knots())

        if knots not in self.tesselations:
            Log.debug('New unique knot vector detected, generating tesselation')
            self.tesselations[knots] = get_tesselation(patch, nvis=self.nvis)
        return self.tesselations[knots]

    def findid(self, patch):
        corners = tuple(tuple(p) for p in patch.corners())
        if corners not in self.corners:
            Log.error('Unable to find corresponding geometry patch')
            return None
        return self.globids[self.corners[corners]]

    def update(self, w, stepid):
        if not self.basis.update_at(stepid):
            return

        Log.info('Updating geometry')

        # FIXME: Here, all patches are updated whenever any of them are updated.
        # Maybe overkill.
        for patchid in range(self.basis.npatches):
            patch = self.basis.patch_at(stepid, patchid)
            key = (self.basis.name, patchid)

            if key not in self.globids:
                self.globids[key] = len(self.globids)
                Log.debug('New unique patch detected, generating global ID')

            # FIXME: This leaves behind invalidated corner IDs, which we should probably delete.
            corners = tuple(tuple(p) for p in patch.corners())
            self.corners[corners] = key
            globpatchid = self.globids[(self.basis.name, patchid)]

            tess = self.tesselation(patch)
            nodes = tess(patch)

            while nodes.shape[-1] < 3:
                nshape = nodes.shape[:-1] + (1,)
                nodes = np.concatenate((nodes, np.zeros(nshape)), axis=-1)

            dim, eidxs = tess.elements()

            Log.debug('Writing patch {}'.format(globpatchid))
            w.update_geometry(nodes, eidxs, dim, globpatchid)

        w.finalize_geometry(stepid)


class Reader:

    def __init__(self, h5, bases=(), geometry=None, nvis=1):
        self.h5 = h5
        self.only_bases = bases
        self.geometry_basis = geometry
        self.nvis = nvis

    def __enter__(self):
        self.bases = OrderedDict()
        self.fields = OrderedDict()

        self.init_bases()

        self.init_fields()
        self.log_fields()
        self.split_fields()
        self.combine_fields()
        self.sort_fields()

        if self.geometry_basis:
            self.geometry = GeometryManager(self.bases[self.geometry_basis], self.nvis)
        else:
            self.geometry = GeometryManager(next(iter(self.bases.values())), self.nvis)

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

        # Delete timeinfo, if present
        if 'timeinfo' in self.bases:
            del self.bases['timeinfo']

        # Delete bases that don't have any patch data
        to_del = [name for name, basis in self.bases.items() if not basis.update_steps]
        for basisname in to_del:
            Log.debug('Basis {} has no updates, removed'.format(basisname))
            del self.bases[basisname]

        # Delete the bases we don't need
        if self.only_bases:
            self.only_bases = tuple(set(self.bases) & set(self.only_bases))
            keep = self.only_bases + ((self.geometry_basis,) if self.geometry_basis else ())
            self.bases = OrderedDict((b,v) for b,v in self.bases.items() if b in keep)
        else:
            self.only_bases = self.bases

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

    def log_fields(self):
        for field in self.fields.values():
            Log.debug('{} "{}" lives on {} ({} components)'.format(
                'Knotspan' if field.cells else 'Field', field.name, field.basis.name, field.ncomps
            ))

    # Detect split fields, e.g. u_y&&T -> u_y, T
    def split_fields(self):
        to_add, to_del = [], []
        for fname in self.fields:
            if '&&' in fname:
                splitnames = [s.strip() for s in fname.split('&&')]

                # Check if the first name has a prefix, if so apply it to all the names
                if ' ' in splitnames[0]:
                    prefix, splitnames[0] = splitnames[0].split(' ', 1)
                    splitnames = ['{} {}'.format(prefix, name) for name in splitnames]

                if any(splitname in self.fields for splitname in splitnames):
                    Log.warning('Unable to split "{}", some fields already exist'.format(fname))
                    continue
                Log.info('Split field "{}" -> {}'.format(fname, ', '.join(splitnames)))
                for i, splitname in enumerate(splitnames):
                    to_add.append(SplitField(splitname, self.fields[fname], i, self))
                to_del.append(fname)

        for fname in to_del:
            del self.fields[fname]
        for field in to_add:
            self.fields[field.name] = field

    # Detect combined fields, e.g. u_x, u_y, u_z -> u
    def combine_fields(self):
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
                Log.info('Creating combined field {} -> {} ({} components)'.format(', '.join(sourcenames), fname, len(sources)))
            except AssertionError:
                Log.warning('Unable to combine {} -> {}'.format(', '.join(sourcenames), fname))

    def sort_fields(self):
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
                    if field.update_at(stepid):
                        Log.info('Updating {} ({})'.format(field.name, field.basisname))
                        field.update(w, stepid)

                w.finalize_step()

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
            raise ValueError('Geometry for basis {} unavailable at timestep {}'.format(basis, level))

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
        field = EigenField('Mode Shape', basis, self, vectorize=True)
        for modeid in range(self.nmodes):
            field.add_update(modeid)
        self.fields['Mode Shape'] = field


def get_reader(filename, bases=(), geometry=None, nvis=1, **kwargs):
    h5 = h5py.File(filename, 'r')
    basisname = next(iter(h5['0']))
    if 'Eigenmode' in h5['0'][basisname]:
        Log.info('Detected eigenmodes')
        return EigenReader(h5, bases=bases, geometry=geometry, nvis=nvis)
    return Reader(h5, bases=bases, geometry=geometry, nvis=nvis)
