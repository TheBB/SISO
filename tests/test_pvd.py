from pathlib import Path
import xml.etree.ElementTree as ET

from click.testing import CliRunner
import numpy as np
import pytest

from .shared import TESTCASES, compare_vtk_unstructured, PreparedTestCase
from ifem_to_vt.__main__ import convert

import vtk


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


@pytest.mark.parametrize('case', TESTCASES['pvd'])
def test_pvd_integrity(case: PreparedTestCase):
    with case.invoke('pvd') as tempdir:
        for out, ref in case.check_files(tempdir):
            compare_pvd(out, ref)
