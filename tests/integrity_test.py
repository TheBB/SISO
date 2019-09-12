from os.path import dirname, join, splitext
import pytest
import tempfile
import re
import numpy as np

from ifem_to_vt.reader import get_reader
from ifem_to_vt.writer import get_writer


NUM = re.compile(r'-?\d+[ +\-e.\d]*\n')
TESTDATA_DIR = join(dirname(__file__), 'testdata')
FILES = [
    'Annulus',
    'Cavity-mixed', 'Cavity3D-compatible',
    'Cyl2D-VMSFSI-weak',
    'singular-pressure-corner-rec',
    'Square', 'Square-ad', 'Square-LR',
    'Square-compatible-abd1-B-I-stat', 'Square-mixed-abd1-B-I-stat',
    'Square-modes', 'Square-modes-freq',
    'Waterfall3D',
]


@pytest.fixture(params=FILES)
def vtf_filenames(request):
    fmt = lambda x: x.format(request.param)
    return join(TESTDATA_DIR, fmt('{}.hdf5')), join(TESTDATA_DIR, fmt('{}.vtf')), fmt('{}.vtf')


@pytest.fixture(params=FILES)
def vtk_filenames(request):
    fmt = lambda x: x.format(request.param)
    return join(TESTDATA_DIR, fmt('{}.hdf5')), join(TESTDATA_DIR, fmt('{}.vtk')), fmt('{}.vtk')


def step_filenames(n, base):
    basename, ext = splitext(base)
    for i in range(1, n+1):
        yield '{}-{}{}'.format(basename, i, ext)


def compare_files(out, ref):
    for outline, refline in zip(out, ref):
        if 'EXPORT_DATE' in outline:
            continue
        if NUM.match(refline):
            out = list(map(float, outline.split()))
            ref = list(map(float, refline.split()))
            np.testing.assert_array_almost_equal(out, ref)
        else:
            assert outline == refline


def test_vtf_integrity(vtf_filenames):
    infile, checkfile, outfile = vtf_filenames
    with tempfile.TemporaryDirectory() as tempdir:
        outfile = join(tempdir, outfile)
        with get_reader(infile) as r, get_writer('vtf')(outfile) as w:
            r.write(w)
        with open(outfile, 'r') as out, open(checkfile, 'r') as ref:
            compare_files(out, ref)


def test_vtk_integrity(vtk_filenames):
    infile, checkfile, outfile = vtk_filenames
    with tempfile.TemporaryDirectory() as tempdir:
        outfile = join(tempdir, outfile)
        with get_reader(infile) as r, get_writer('vtk')(outfile) as w:
            nsteps = r.nsteps
            r.write(w)
        for outfn, checkfn in zip(step_filenames(nsteps, outfile), step_filenames(nsteps, checkfile)):
            with open(outfn, 'r') as out, open(checkfn, 'r') as ref:
                compare_files(out, ref)
