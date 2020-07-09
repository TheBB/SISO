from os.path import join, splitext, dirname
import pytest
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .shared import TESTDATA_DIR, FILES, step_filenames, compare_vtk_unstructured

from ifem_to_vt import Config
from ifem_to_vt.reader import get_reader
from ifem_to_vt.writer import get_writer

import vtk
has_vtk_9 = vtk.vtkVersion().GetVTKMajorVersion() >= 9


@pytest.fixture(params=FILES)
def filenames(request):
    rootdir, rootname = request.param
    basename, _ = splitext(rootname)
    pvdname = '{}.pvd'.format(basename)
    return (
        join(TESTDATA_DIR, rootdir, rootname),
        join(TESTDATA_DIR, 'pvd', pvdname),
        pvdname,
    )


def load_grid(filename):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def compare_pvd(outfn, reffn):
    with open(outfn, 'r') as f:
        out = ET.fromstring(f.read())
    assert out.tag == 'VTKFile'
    assert out.attrib['type'] == 'Collection'
    assert len(out) == 1
    out = next(iter(out))
    assert out.tag == 'Collection'

    with open(reffn, 'r') as f:
        ref = ET.fromstring(f.read())
    ref = next(iter(ref))

    assert len(out) == len(ref)
    for outtag, reftag in zip(out, ref):
        assert outtag.tag == 'DataSet'
        np.testing.assert_allclose(float(outtag.attrib['timestep']), float(reftag.attrib['timestep']))
        assert outtag.attrib['part'] == reftag.attrib['part']
        compare_vtk_unstructured(
            load_grid(join(dirname(outfn), outtag.attrib['file'])),
            load_grid(join(dirname(reffn), reftag.attrib['file'])),
        )


@pytest.mark.skipif(not has_vtk_9, reason="VTK tests only work on VTK>=9")
def test_pvd_integrity(filenames):
    infile, checkfile, outfile = filenames
    with tempfile.TemporaryDirectory() as tempdir:
        outfile = join(tempdir, outfile)
        cfg = Config(mode='ascii')
        with get_reader(infile, cfg) as r, get_writer('pvd')(outfile, cfg) as w:
            r.write(w)
        compare_pvd(outfile, checkfile)
