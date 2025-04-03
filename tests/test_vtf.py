from __future__ import annotations

import numpy as np
import pytest

from .shared import TESTCASES, TESTIDS, PreparedTestCase

try:
    import vtfwriter as vtf

    has_vtf = True
except ImportError:
    has_vtf = False


def compare_stateinfo(out, ref):
    o = out.GetBlockByType(vtf.STATEINFO, 0)
    r = out.GetBlockByType(vtf.STATEINFO, 0)
    assert o.GetNumStateInfos() == r.GetNumStateInfos()
    for i in range(o.GetNumStateInfos()):
        assert o.GetStepName(i) == r.GetStepName(i)
        assert o.GetStepRefValue(i) == r.GetStepRefValue(i)
        assert o.GetStepRefType(i) == r.GetStepRefType(i)


def compare_result(out, ref, case):
    assert out.GetDimension() == ref.GetDimension()
    assert out.GetResultMapping() == ref.GetResultMapping()
    np.testing.assert_allclose(
        out.GetResults(),
        ref.GetResults(),
        atol=case.abs_tol,
        rtol=case.rel_tol,
    )


def compare_field(out, ref, case):
    assert out.GetName() == ref.GetName()
    assert out.GetNumSteps() == ref.GetNumSteps()
    for i in range(out.GetNumSteps()):
        assert out.GetStepNumber(i) == ref.GetStepNumber(i)
        assert out.GetNumResultBlocks(i) == ref.GetNumResultBlocks(i)
        out_results = out.GetResultBlocks(i)
        ref_results = ref.GetResultBlocks(i)
        assert len(out_results) == out.GetNumResultBlocks(i)
        assert len(ref_results) == out.GetNumResultBlocks(i)
        for o, r in zip(out_results, ref_results):
            compare_result(o, r, case)


def compare_fields(out, ref, tp, case):
    out_fields = [out.GetBlockByType(tp, i) for i in range(out.GetNumBlocksByType(tp))]
    ref_fields = [ref.GetBlockByType(tp, i) for i in range(ref.GetNumBlocksByType(tp))]
    assert len(out_fields) == len(ref_fields)
    out_byname = {f.GetName(): f for f in out_fields}
    ref_byname = {f.GetName(): f for f in ref_fields}
    assert len(out_byname) == len(out_fields)
    assert len(ref_byname) == len(ref_fields)
    assert set(out_byname.keys()) == set(ref_byname.keys())
    for name in out_byname:
        compare_field(out_byname[name], ref_byname[name], case)


def compare_eblock(out, ref, case):
    assert out.GetPartID() == ref.GetPartID()
    assert out.GetPartName() == ref.GetPartName()
    assert out.GetNumElementGroups() == ref.GetNumElementGroups()
    for i in range(out.GetNumElementGroups()):
        otp, oids = out.GetElementGroup(i)
        rtp, rids = ref.GetElementGroup(i)
        assert otp == rtp
        assert (oids == rids).all()
    onodes = out.GetNodeBlock().GetNodes()
    rnodes = ref.GetNodeBlock().GetNodes()
    np.testing.assert_allclose(
        onodes,
        rnodes,
        atol=case.abs_tol,
        rtol=case.rel_tol,
    )


def compare_geometry(out, ref, case):
    o = out.GetBlockByType(vtf.GEOMETRY, 0)
    r = ref.GetBlockByType(vtf.GEOMETRY, 0)
    assert o.GetNumSteps() == r.GetNumSteps()
    for i in range(o.GetNumSteps()):
        assert o.GetStepNumber(i) == r.GetStepNumber(i)
        out_blocks = o.GetElementBlocks(i)
        ref_blocks = r.GetElementBlocks(i)
        assert len(out_blocks) == len(ref_blocks)
        for ob, rb in zip(out_blocks, ref_blocks):
            compare_eblock(ob, rb, case)


def compare_vtf(out, ref, case: PreparedTestCase):
    compare_stateinfo(out, ref)
    compare_fields(out, ref, vtf.SCALAR, case)
    compare_fields(out, ref, vtf.VECTOR, case)
    compare_fields(out, ref, vtf.DISPLACEMENT, case)
    compare_geometry(out, ref, case)


@pytest.mark.skipif(not has_vtf, reason="VTF tests not runnable without vtfwriter")
@pytest.mark.parametrize("case", TESTCASES["vtf"], ids=TESTIDS["vtf"])
def test_vtf_integrity(case: PreparedTestCase):
    with case.invoke("vtf") as tempdir:
        for out, ref in case.check_files(tempdir):
            assert out.exists()
            assert ref.exists()
            with vtf.File(str(out), "r") as o, vtf.File(str(ref), "r") as r:
                compare_vtf(o, r, case)
