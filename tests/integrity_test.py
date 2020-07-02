from os.path import dirname, join, splitext
import pytest
import tempfile
import re
import numpy as np
import xml.etree.ElementTree as ElementTree

from ifem_to_vt.reader import get_reader
from ifem_to_vt.writer import get_writer

try:
    import vtfwriter
    has_vtf = True
except ImportError:
    has_vtf = False


from vtk import vtkVersion
has_vtk_9 = vtkVersion().GetVTKMajorVersion() >= 9


NUM = re.compile(r'-?\d+[ +\-e.\d]*\n?$')
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


@pytest.fixture(params=FILES)
def vtu_filenames(request):
    fmt = lambda x: x.format(request.param)
    return join(TESTDATA_DIR, fmt('{}.hdf5')), join(TESTDATA_DIR, fmt('{}.vtu')), fmt('{}.vtu')


def step_filenames(n, base):
    basename, ext = splitext(base)
    for i in range(1, n+1):
        yield '{}-{}{}'.format(basename, i, ext)


def compare_xml_elements(out, ref):
    assert out.tag == ref.tag
    assert len(out) >= len(ref)
    if out.tag == 'DataArray':
        outd = list(map(float, out.text.split()))
        refd = list(map(float, ref.text.split()))
        np.testing.assert_allclose(outd, refd, atol=1e-10)
    assert set(out.attrib) == set(ref.attrib)
    for key in out.attrib:
        if NUM.match(out.attrib[key]):
            outv = list(map(float, out.attrib[key].split()))
            refv = list(map(float, ref.attrib[key].split()))
            np.testing.assert_allclose(outv, refv, atol=1e-10)
        else:
            assert out.attrib[key] == ref.attrib[key]
    for outc, refc in zip(out, ref):
        compare_xml_elements(outc, refc)


def compare_xml_files(out, ref):
    out = ElementTree.fromstring(out.read())
    ref = ElementTree.fromstring(ref.read())
    compare_xml_elements(out, ref)


def compare_files(out, ref):
    for outline, refline in zip(out, ref):
        if 'EXPORT_DATE' in outline:
            continue
        if NUM.match(refline):
            out = list(map(float, outline.split()))
            ref = list(map(float, refline.split()))
            np.testing.assert_allclose(out, ref, atol=1e-10)
        else:
            assert outline == refline


@pytest.mark.skipif(not has_vtf, reason="VTF tests not runnable without vtfwriter")
def test_vtf_integrity(vtf_filenames):
    infile, checkfile, outfile = vtf_filenames
    with tempfile.TemporaryDirectory() as tempdir:
        outfile = join(tempdir, outfile)
        with get_reader(infile) as r, get_writer('vtf')(outfile, mode='ascii') as w:
            r.write(w)
        with open(outfile, 'r') as out, open(checkfile, 'r') as ref:
            compare_files(out, ref)


@pytest.mark.skipif(not has_vtk_9, reason="VTK tests only work on VTK>=9")
def test_vtk_integrity(vtk_filenames):
    infile, checkfile, outfile = vtk_filenames
    with tempfile.TemporaryDirectory() as tempdir:
        outfile = join(tempdir, outfile)
        with get_reader(infile) as r, get_writer('vtk')(outfile, mode='ascii') as w:
            nsteps = r.nsteps
            r.write(w)
        for outfn, checkfn in zip(step_filenames(nsteps, outfile), step_filenames(nsteps, checkfile)):
            with open(outfn, 'r') as out, open(checkfn, 'r') as ref:
                compare_files(out, ref)


def test_vtu_integrity(vtu_filenames):
    infile, checkfile, outfile = vtu_filenames
    with tempfile.TemporaryDirectory() as tempdir:
        outfile = join(tempdir, outfile)
        with get_reader(infile) as r, get_writer('vtu')(outfile, mode='ascii') as w:
            nsteps = r.nsteps
            r.write(w)
        for outfn, checkfn in zip(step_filenames(nsteps, outfile), step_filenames(nsteps, checkfile)):
            with open(outfn, 'r') as out, open(checkfn, 'r') as ref:
                compare_xml_files(out, ref)
