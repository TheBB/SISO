from pathlib import Path
import pytest

from click.testing import CliRunner

from .shared import TESTCASES, compare_vtk_unstructured, PreparedTestCase

import vtk


def load_grid(filename: Path):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(str(filename))
    reader.Update()
    return reader.GetOutput()


def compare_vtk(out: Path, ref: Path, case: PreparedTestCase):
    assert out.exists()
    assert ref.exists()
    compare_vtk_unstructured(load_grid(str(out)), load_grid(ref), case)


@pytest.mark.parametrize('case', TESTCASES['vtk'])
def test_vtk_integrity(case: PreparedTestCase):
    with case.invoke('vtk') as tempdir:
        for out, ref in case.check_files(tempdir):
            compare_vtk(out, ref, case)
