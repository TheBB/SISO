from pathlib import Path
import pytest

from click.testing import CliRunner

from .shared import TESTCASES, TESTIDS, compare_vtk_structured, PreparedTestCase

import vtk


def load_grid(filename: Path):
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(str(filename))
    reader.Update()
    return reader.GetOutput()


def compare_vts(out: Path, ref: Path, case: PreparedTestCase):
    assert out.exists()
    assert ref.exists()
    compare_vtk_structured(load_grid(out), load_grid(ref), case)


@pytest.mark.parametrize('case', TESTCASES['vts'], ids=TESTIDS['vts'])
def test_vts_integrity(case: PreparedTestCase):
    with case.invoke('vts') as tempdir:
        for out, ref in case.check_files(tempdir):
            compare_vts(out, ref, case)
