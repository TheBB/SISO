from os.path import splitext, basename
from pathlib import Path
import pytest
import xml.etree.ElementTree as ET

import numpy as np
from click.testing import CliRunner

from .shared import TESTDATA_DIR, FILES, step_filenames, compare_vtk_unstructured, cd_temp
from ifem_to_vt.__main__ import convert

import vtk
has_vtk_9 = vtk.vtkVersion().GetVTKMajorVersion() >= 9


@pytest.fixture(params=FILES)
def filenames(request):
    rootdir, rootname, nsteps = request.param
    base, _ = splitext(basename(rootname))
    pvdname = '{}.pvd'.format(base)
    return (
        TESTDATA_DIR / rootdir / rootname,
        TESTDATA_DIR / 'pvd' / pvdname,
        pvdname,
        nsteps,
    )


def load_grid(filename: Path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(filename))
    reader.Update()
    return reader.GetOutput()


def compare_pvd(outfn: Path, reffn: Path):
    assert outfn.exists()
    assert reffn.exists()
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
            load_grid(outfn.parent / outtag.attrib['file']),
            load_grid(reffn.parent / reftag.attrib['file']),
        )


@pytest.mark.skipif(not has_vtk_9, reason="VTK tests only work on VTK>=9")
def test_pvd_integrity(filenames):
    infile, checkfile, outfile, nsteps = filenames
    with cd_temp() as tempdir:
        outfile = tempdir / outfile
        res = CliRunner().invoke(convert, ['--mode', 'ascii', '-f', 'pvd', str(infile)])
        assert res.exit_code == 0
        compare_pvd(outfile, checkfile)
