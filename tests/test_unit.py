from __future__ import annotations

from itertools import islice

from siso.filter.timeslice import islice_flag


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
