from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import vtk

from .shared import TESTCASES, TESTIDS, PreparedTestCase, compare_vtk_unstructured

if TYPE_CHECKING:
    from pathlib import Path


def load_grid(filename: Path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(filename))
    reader.Update()
    return reader.GetOutput()


def compare_vtu(out: Path, ref: Path, case: PreparedTestCase):
    assert out.exists()
    assert ref.exists()
    compare_vtk_unstructured(load_grid(out), load_grid(ref), case)


@pytest.mark.parametrize("case", TESTCASES["vtu"], ids=TESTIDS["vtu"])
def test_vtu_integrity(case: PreparedTestCase):
    with case.invoke("vtu") as tempdir:
        for out, ref in case.check_files(tempdir):
            compare_vtu(out, ref, case)
