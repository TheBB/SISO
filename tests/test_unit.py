from __future__ import annotations

from itertools import islice

import numpy as np

from siso import api
from siso.filter.timeslice import islice_flag
from siso.impl import Field
from siso.reader.ifem import MaskedTopology
from siso.topology import CellType, StructuredTopology
from siso.util import FieldData


def my_islice(it, *args):
    for v, f in zip(it, islice_flag(*args)):
        if f:
            yield v


def test_islice():
    def theirs(*args):
        return list(islice("abcdefghijklmnopqrstuvwxyz", *args))

    def mine(*args):
        return list(my_islice("abcdefghijklmnopqrstuvwxyz", *args))

    assert theirs(3) == mine(3) == list("abc")
    assert theirs(None) == mine(None) == list("abcdefghijklmnopqrstuvwxyz")
    assert theirs(1, 4) == mine(1, 4) == list("bcd")
    assert theirs(None, 4) == mine(None, 4) == list("abcd")
    assert theirs(3, None) == mine(3, None) == list("defghijklmnopqrstuvwxyz")
    assert theirs(None, None) == mine(None, None) == list("abcdefghijklmnopqrstuvwxyz")
    assert theirs(1, 10, 2) == mine(1, 10, 2) == list("bdfhj")
    assert theirs(None, 10, 2) == mine(None, 10, 2) == list("acegi")
    assert theirs(1, None, 2) == mine(1, None, 2) == list("bdfhjlnprtvxz")
    assert theirs(1, 10, None) == mine(1, 10, None) == list("bcdefghij")
    assert theirs(None, None, 2) == mine(None, None, 2) == list("acegikmoqsuwy")
    assert theirs(None, 5, None) == mine(None, 5, None) == list("abcde")
    assert theirs(5, None, None) == mine(5, None, None) == list("fghijklmnopqrstuvwxyz")
    assert theirs(None, None, None) == mine(None, None, None) == list("abcdefghijklmnopqrstuvwxyz")


def test_ifem_element_activation_mask():
    topology = StructuredTopology(api.CellShape((2, 1)), CellType.Quadrilateral, degree=1)
    masked = MaskedTopology(topology, FieldData(np.array([[0.0], [1.0]])))
    discrete, mapper = masked.discretize(1)

    nodal_field = Field("nodal", api.Scalar())
    cell_field = Field("cell", api.Scalar(), cellwise=True)

    assert discrete.num_cells == 1
    assert discrete.num_nodes == 4
    np.testing.assert_array_equal(mapper(cell_field, FieldData(np.array([[10.0], [20.0]]))).numpy(), [[20.0]])
    np.testing.assert_array_equal(
        mapper(nodal_field, FieldData(np.arange(12.0).reshape(6, 2))).numpy(),
        [[4.0, 5.0], [6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
    )

    merger = masked.create_merger()
    merged, merge_mapper = merger(masked)
    assert merged.num_cells == 1
    assert merged.num_nodes == 4
    np.testing.assert_array_equal(
        merge_mapper(cell_field, FieldData(np.array([[10.0], [20.0]]))).numpy(), [[20.0]]
    )
