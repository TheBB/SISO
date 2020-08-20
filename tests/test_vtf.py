from pathlib import Path
import pytest
import re

import numpy as np
from click.testing import CliRunner

from .shared import TESTCASES, PreparedTestCase

try:
    import vtfwriter
    has_vtf = True
except ImportError:
    has_vtf = False


NUM = re.compile(r'-?\d+[ +\-e.\d]*\n?$')


def compare_vtf(out, ref):
    outiter = iter(out)
    refiter = iter(ref)
    for i, (outline, refline) in enumerate(zip(outiter, refiter), start=1):
        if 'EXPORT_DATE' in outline:
            continue
        if NUM.match(refline):
            out = list(map(float, outline.split()))
            ref = list(map(float, refline.split()))
            np.testing.assert_allclose(out, ref, atol=1e-10)
        else:
            assert outline == refline

    # Check that files are equally long
    assert not list(outiter)
    assert not list(refiter)


@pytest.mark.skipif(not has_vtf, reason="VTF tests not runnable without vtfwriter")
@pytest.mark.parametrize('case', TESTCASES['vtf'])
def test_vtf_integrity(case: PreparedTestCase):
    with case.invoke('vtf') as tempdir:
        for out, ref in case.check_files(tempdir):
            assert out.exists()
            assert ref.exists()
            with open(out, 'r') as a, open(ref, 'r') as b:
                compare_vtf(a, b)
