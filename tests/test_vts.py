from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import vtk

from .shared import TESTCASES, TESTIDS, PreparedTestCase, compare_vtk_structured

if TYPE_CHECKING:
    from pathlib import Path


def load_grid(filename: Path):
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(str(filename))
    reader.Update()
    return reader.GetOutput()


def compare_vts(out: Path, ref: Path, case: PreparedTestCase):
    assert out.exists()
    assert ref.exists()
    compare_vtk_structured(load_grid(out), load_grid(ref), case)


@pytest.mark.parametrize("case", TESTCASES["vts"], ids=TESTIDS["vts"])
def test_vts_integrity(case: PreparedTestCase):
    with case.invoke("vts") as tempdir:
        for out, ref in case.check_files(tempdir):
            compare_vts(out, ref, case)
