import h5py
from contextlib import contextmanager
from collections import namedtuple, OrderedDict
from itertools import chain, product
import lrspline as lr
import numpy as np
import splipy.io
from splipy import SplineObject, BSplineBasis
from splipy.SplineModel import ObjectCatalogue
import splipy.utils
import sys
import treelog as log

from ..geometry import SplinePatch


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
                patch = SplinePatch(g2bytes)
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

            num = len(coeffs)
            denom = patch.num_cells if self.cells else patch.num_nodes

            if num % denom != 0:
                log.error('Inconsistent dimension in field "{}" ({}/{}); unable to discover number of components'.format(
                    self.name, num, denom
                ))
                sys.exit(2)
            self.ncomps = num // denom

    def update_at(self, stepid):
        return stepid in self.update_steps

    def update(self, w, stepid):
        for patchid in range(self.basis.npatches):
            patch = self.basis.patch_at(stepid, patchid)
            globpatchid = self.reader.geometry.findid(patch)
            if globpatchid is None:
                continue

            coeffs = self.coeffs(stepid, patchid)

            results = self.tesselate(patch, coeffs)
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

    def tesselate(self, patch, coeffs):
        # TODO: nvis
        results = patch.tesselate_coeffs(coeffs, cells=self.cells, nvis=1)

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
                globpatchid = self.reader.geometry.findid(patch)
                coeffs = src.coeffs(stepid, patchid)
                results.append(src.tesselate(patch, coeffs))

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
        self.has_updated = False

        # Map knot vector -> evaluation points
        self.tesselations = {}

        # Map basisname x patchid ->
        self.globids = {}

        # Map corner tuple -> basisname x patchid
        self.corners = {}

        log.info('Using {} for geometry'.format(basis.name))

    def findid(self, patch):
        corners = patch.key
        if corners not in self.corners:
            log.error('Unable to find corresponding geometry patch')
            return None
        return self.globids[self.corners[corners]]

    def update(self, w, stepid):
        if self.has_updated and not self.basis.update_at(stepid):
            return
        self.has_updated = True

        log.info('Updating geometry')

        # FIXME: Here, all patches are updated whenever any of them are updated.
        # Maybe overkill.
        for patchid in range(self.basis.npatches):
            patch = self.basis.patch_at(stepid, patchid)
            key = (self.basis.name, patchid)

            if key not in self.globids:
                self.globids[key] = len(self.globids)
                log.debug('New unique patch detected, generating global ID')

            # FIXME: This leaves behind invalidated corner IDs, which we should probably delete.
            self.corners[patch.key] = key
            globpatchid = self.globids[(self.basis.name, patchid)]

            tess = patch.tesselate(nvis=self.nvis)
            nodes = tess.nodes
            dim = 2 if tess.cells.shape[-1] == 4 else 3
            eidxs = tess.cells

            while nodes.shape[-1] < 3:
                nshape = nodes.shape[:-1] + (1,)
                nodes = np.concatenate((nodes, np.zeros(nshape)), axis=-1)

            log.debug('Writing patch {}'.format(globpatchid))
            w.update_geometry(nodes, eidxs, dim, globpatchid)

        w.finalize_geometry(stepid)


class Reader:

    def __init__(self, h5, bases=(), geometry=None, nvis=1, last=False, **kwargs):
        self.h5 = h5
        self.only_bases = bases
        self.geometry_basis = geometry
        self.nvis = nvis
        self.last = last

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
            try:
                time = self.h5[str(stepid)]['timeinfo']['level'][0]
            except KeyError:
                time = float(stepid)
            yield stepid, {'time': time}, self.h5[str(stepid)]

    def outputsteps(self):
        if self.last:
            for stepid, time, _ in self.steps():
                pass
            yield stepid, time
        else:
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
            log.debug('Basis {} has no updates, removed'.format(basisname))
            del self.bases[basisname]

        # Delete the bases we don't need
        if self.only_bases:
            self.only_bases = tuple(set(self.bases) & set(self.only_bases))
            keep = self.only_bases + ((self.geometry_basis,) if self.geometry_basis else ())
            self.bases = OrderedDict((b,v) for b,v in self.bases.items() if b in keep)
        else:
            self.only_bases = self.bases

        for basis in self.bases.values():
            log.debug('Basis {} updates at steps {} ({} patches)'.format(
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
            log.debug('{} "{}" lives on {} ({} components)'.format(
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
                    log.warning('Unable to split "{}", some fields already exist'.format(fname))
                    continue
                log.info('Split field "{}" -> {}'.format(fname, ', '.join(splitnames)))
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
                log.info('Creating combined field {} -> {} ({} components)'.format(', '.join(sourcenames), fname, len(sources)))
            except AssertionError:
                log.warning('Unable to combine {} -> {}'.format(', '.join(sourcenames), fname))

    def sort_fields(self):
        fields = sorted(self.fields.values(), key=lambda f: f.name)
        fields = sorted(fields, key=lambda f: f.cells)
        self.fields = OrderedDict((f.name, f) for f in fields)

    def write(self, w):
        for stepid, time in log.iter.plain('Step', self.outputsteps()):
            w.add_step(**time)
            self.geometry.update(w, stepid)

            for field in self.fields.values():
                if field.update_at(stepid):
                    log.info('Updating {} ({})'.format(field.name, field.basisname))
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


def get_reader(filename, **kwargs):
    h5 = h5py.File(filename, 'r')
    basisname = next(iter(h5['0']))
    if 'Eigenmode' in h5['0'][basisname]:
        log.info('Detected eigenmodes')
        return EigenReader(h5, **kwargs)
    return Reader(h5, **kwargs)
