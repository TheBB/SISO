from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.io import FortranFile

from .shared import TESTCASES, TESTIDS, PreparedTestCase

if TYPE_CHECKING:
    from pathlib import Path


def to_native(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.byteorder in ("=", sys.byteorder):
        return arr
    return arr.byteswap().newbyteorder()


def compare_simra(out: Path, ref: Path, case: PreparedTestCase):
    out_u4 = np.dtype(f"{case.out_endian}u4")
    out_f4 = np.dtype(f"{case.out_endian}f4")
    ref_u4 = np.dtype(f"{case.ref_endian}u4")
    ref_f4 = np.dtype(f"{case.ref_endian}f4")

    with FortranFile(out, header_dtype=out_u4) as o, FortranFile(ref, header_dtype=ref_u4) as r:
        out_header = to_native(o.read_ints(out_u4))
        ref_header = to_native(r.read_ints(ref_u4))
        assert (out_header == ref_header).all()
        out_nodes = to_native(o.read_reals(out_f4))
        ref_nodes = to_native(r.read_reals(ref_f4))
        assert (out_nodes == ref_nodes).all()
        out_cells = to_native(o.read_ints(out_u4))
        ref_cells = to_native(r.read_ints(ref_u4))
        assert (out_cells == ref_cells).all()
        out_mcells = to_native(o.read_ints(out_u4))
        ref_mcells = to_native(r.read_ints(ref_u4))
        assert (out_mcells == ref_mcells).all()


@pytest.mark.parametrize("case", TESTCASES["simra"], ids=TESTIDS["simra"])
def test_simra_integrity(case: PreparedTestCase):
    with case.invoke("simra") as tempdir:
        for out, ref in case.check_files(tempdir):
            compare_simra(out, ref, case)
