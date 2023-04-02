from __future__ import annotations

from contextlib import contextmanager
from functools import reduce
from itertools import chain, count, product
from pathlib import Path
from typing import (
    IO,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Generic,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
    runtime_checkable,
)

import lrspline as lr
import numpy as np
from numpy import integer
from typing_extensions import Self

from .. import api
from .field_data import FieldData


@runtime_checkable
class HasName(Protocol):
    name: ClassVar[str]


W = TypeVar("W")
M = TypeVar("M", bound=Hashable)
Q = TypeVar("Q", bound=HasName)


class Registry(Generic[W]):
    classes: Dict[str, W]

    def __init__(self):
        self.classes = {}

    @overload
    def register(self, arg: str) -> Callable[[W], W]:
        ...

    @overload
    def register(self, arg: Type[Q]) -> Type[Q]:
        ...

    def register(self, arg):
        if not isinstance(arg, str):
            assert isinstance(arg, HasName)
            self.classes[arg.name.casefold()] = arg
            return arg

        def decorator(x: W) -> W:
            self.classes[arg] = x
            return x

        return decorator

    def __getitem__(self, key: str) -> W:
        return self.classes[key.casefold()]

    def __contains__(self, key: str) -> bool:
        return key.casefold() in self.classes

    def items(self) -> Iterable[Tuple[str, W]]:
        return self.classes.items()


class NoSuchMarkError(Exception):
    ...


class RandomAccessFile(Generic[W, M]):
    """Utility class for wrapping a file pointer in an interface that allows
    random access. The file pointer in question must be seekable.

    This class has the ability to track 'markers': named locations in the file
    that can be returned to. They can be produced on demand by a marker
    generator: a function that generates marks as it finds them.

    Explicit access to the file is only granted to one user at a time, through
    use of the `borrow_fp` or `leap` context managers, or the
    `RandomAccessTracker` class (see below).

    Parameters:
    - fp: the file pointer to wrap
    - wrapper: callable for constructing 'wrapper' objects from the file pointer
    - marker_generator: a function that accepts a `RandomAccessTracker` and
        generates marks. The function will always receive the same instance of
        tracker, so it can be used to remember file location.
    """

    wrapper: Callable[[IO], W]

    markers: Dict[M, int]
    marker_generator: Optional[Iterator[Tuple[M, int]]] = None

    fp: IO
    fp_borrowed: bool = False

    def __init__(
        self,
        fp: IO,
        wrapper: Optional[Callable[[IO], W]] = None,
        marker_generator: Optional[Callable[[RandomAccessTracker[W, M]], Iterator[Tuple[M, int]]]] = None,
    ):
        assert fp.seekable()
        self.fp = fp
        self.wrapper = wrapper or cast(Callable[[IO], W], (lambda fp: fp))
        self.markers = {}

        if marker_generator:
            self.marker_generator = marker_generator(self.tracker())

    def __enter__(self) -> Self:
        self.fp = self.fp.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self.fp.__exit__(*args)

    @contextmanager
    def borrow_fp(self) -> Generator[IO, None, None]:
        """Borrow the file pointer. This will error if the file pointer is
        already borrowed: only one borrower can use it at a time.

        User code should generally use `leap` instead of this method, if
        possible.

        Use as a context manager to release the file pointer after use:

        ```
        with file.borrow_fp(mark) as fp:
            ...
        ```
        """
        if self.fp_borrowed:
            raise api.Unexpected("Borrowing already borrowed file pointer")
        assert not self.fp_borrowed
        self.fp_borrowed = True
        try:
            yield self.fp
        finally:
            self.fp_borrowed = False

    def mark(self, name: M, loc: Optional[int] = None) -> None:
        """Mark the location at `loc` with the name `name`. If `loc` is not
        given, this will use the current position of the file pointer.
        """
        if loc is None:
            loc = self.fp.tell()
        self.markers[name] = loc

    def loc_at(self, name: M) -> int:
        """Return the offset at the given mark.

        If the mark has not been discovered yet, but a marker generator was
        provided at construction time, this will attempt to generate marks until
        the requested mark is found.
        """
        if name in self.markers:
            return self.markers[name]
        if self.marker_generator:
            for mark, loc in self.marker_generator:
                self.mark(mark, loc)
                if mark == name:
                    return loc
        raise NoSuchMarkError(name)

    @contextmanager
    def leap(self, mark: M) -> Generator[W, None, None]:
        """Borrow the file pointer, set its location to the requested mark, and
        return it wrapped in the type W.

        Use as a context manager to release the file pointer after use:

        ```
        with file.leap(mark) as w:
            ...
        ```
        """
        loc = self.loc_at(mark)
        with self.borrow_fp() as fp:
            fp.seek(loc)
            yield self.wrapper(fp)

    def tracker(self, mark: Optional[M] = None) -> RandomAccessTracker[W, M]:
        """Create a new tracker object initialized at a given mark (or at the
        beginning of the file, if not given).

        See `RandomAccessTracker` for more info.
        """
        if mark is None:
            return RandomAccessTracker(self, 0)
        return RandomAccessTracker(self, self.loc_at(mark))


class RandomAccessTracker(Generic[W, M]):
    """A tracking object for use with `RandomAccessFile`.

    The tracker remembers the last location that it was used with, and can be
    used to restart from there. Trackers don't own a copy of the file pointer,
    so multiple of them can exist at the same time. Use one of the `excursion`
    or `journey` methods to access the file.
    """

    file: RandomAccessFile[W, M]
    continue_from: int

    def __init__(self, file: RandomAccessFile[W, M], loc: int):
        self.file = file
        self.continue_from = loc

    @contextmanager
    def excursion(self) -> Generator[W, None, None]:
        """Borrow the file pointer and return it wrapped in the type W,
        starting at the previously abandoned location of the tracker.

        An excursion does not change the location of the tracker. After the
        excursion is over, the tracker will restart from the previous location
        in the file.

        Use as a context manager to release the file pointer after use:

        ```
        with tracker.excursion() as w:
            ...
        ```
        """
        with self.file.borrow_fp() as fp:
            fp.seek(self.continue_from)
            yield self.file.wrapper(fp)

    @contextmanager
    def journey(self) -> Generator[W, None, None]:
        """Borrow the file pointer and return it wrapped in the type W,
        starting at the previously abandoned location of the tracker.

        A journey updates the location of the tracker, so that after the journey
        is over, the tracker will restart from the location where the journey
        ended.
        """
        with self.file.borrow_fp() as fp:
            fp.seek(self.continue_from)
            try:
                yield self.file.wrapper(fp)
            finally:
                self.continue_from = fp.tell()

    def origin_marker(self, name: M) -> Tuple[M, int]:
        """Create a tuple of mark and location.

        Use with marker generators (see `RandomAccessFile`).
        """
        return (name, self.continue_from)


@contextmanager
def save_excursion(fp: IO):
    if not fp.seekable():
        raise api.Unsupported("Siso requires seekable file handles")
    ptr = fp.tell()
    try:
        yield
    finally:
        fp.seek(ptr)


def pluralize(num: int, singular: str, plural: str) -> str:
    return f"{num} {singular if num == 1 else plural}"


def flatten_2d(array: np.ndarray) -> np.ndarray:
    if array.ndim == 1:
        return array[:, np.newaxis]
    return array.reshape(-1, array.shape[-1])


def transpose_butlast(array: np.ndarray) -> np.ndarray:
    last = array.ndim - 1
    permutation = tuple(reversed(range(last))) + (last,)
    return array.transpose(permutation)


def _single_slice(ndims: int, axis: int, *args) -> Tuple[slice, ...]:
    retval = [slice(None)] * ndims
    retval[axis] = slice(*args)
    return tuple(retval)


def _single_index(ndims: int, axis: int, ix: int) -> Tuple[Union[slice, int], ...]:
    retval: List[Union[slice, int]] = [slice(None)] * ndims
    retval[axis] = ix
    return tuple(retval)


def _expand_shape(shape: Tuple[int, ...], axis: int) -> Tuple[int, ...]:
    retval = list(shape)
    retval[axis] += 1
    return tuple(retval)


def unstagger(data: np.ndarray, axis: int) -> np.ndarray:
    return (data[_single_slice(data.ndim, axis, 1, None)] + data[_single_slice(data.ndim, axis, 0, -1)]) / 2


def stagger(data: np.ndarray, axis: int) -> np.ndarray:
    retval = np.zeros_like(data, shape=_expand_shape(data.shape, axis))

    first = _single_slice(data.ndim, axis, 0, 1)
    second = _single_slice(data.ndim, axis, 1, 2)
    penultimate = _single_slice(data.ndim, axis, -2, -1)
    last = _single_slice(data.ndim, axis, -1, None)

    retval[_single_slice(data.ndim, axis, 0, -1)] += data / 2
    retval[_single_slice(data.ndim, axis, 1, None)] += data / 2
    retval[first] += data[first] - data[second] / 2
    retval[last] += data[last] - data[penultimate] / 2

    return retval


T = TypeVar("T")


def pairwise(iterable: Iterable[T]) -> Iterator[Tuple[T, T]]:
    it = iter(iterable)
    left = next(it)
    for right in it:
        yield left, right
        left = right


def subdivide_linear(knots: Union[List[float], Tuple[float, ...]], nvis: int) -> np.ndarray:
    return np.fromiter(
        chain(
            chain.from_iterable(
                (((nvis - i) * a + i * b) / nvis for i in range(nvis)) for a, b in pairwise(knots)
            ),
            (knots[-1],),
        ),
        float,
    )


def visit_face(
    element: lr.Element, nodes: Dict[Tuple[float, ...], int], elements: List[List[int]], nvis: int
) -> None:
    lft, btm = element.start()
    rgt, top = element.end()
    xs = subdivide_linear((lft, rgt), nvis)
    ys = subdivide_linear((btm, top), nvis)

    for lft, rgt in pairwise(xs):
        for btm, top in pairwise(ys):
            sw, se, nw, ne = (lft, btm), (rgt, btm), (lft, top), (rgt, top)
            for pt in (sw, se, nw, ne):
                nodes.setdefault(pt, len(nodes))
            elements.append([nodes[sw], nodes[nw], nodes[se], nodes[ne]])


def visit_volume(
    element: lr.Element, nodes: Dict[Tuple[float, ...], int], elements: List[List[int]], nvis: int
) -> None:
    umin, vmin, wmin = element.start()
    umax, vmax, wmax = element.end()
    us = subdivide_linear((umin, umax), nvis)
    vs = subdivide_linear((vmin, vmax), nvis)
    ws = subdivide_linear((wmin, wmax), nvis)

    for ul, ur in pairwise(us):
        for vl, vr in pairwise(vs):
            for wl, wr in pairwise(ws):
                bsw, bse, bnw, bne = (ul, vl, wl), (ur, vl, wl), (ul, vr, wl), (ur, vr, wl)
                tsw, tse, tnw, tne = (ul, vl, wr), (ur, vl, wr), (ul, vr, wr), (ur, vr, wr)
                for pt in (bsw, bse, bnw, bne, tsw, tse, tnw, tne):
                    nodes.setdefault(pt, len(nodes))
                elements.append(
                    [
                        nodes[bsw],
                        nodes[tsw],
                        nodes[bnw],
                        nodes[tnw],
                        nodes[bse],
                        nodes[tse],
                        nodes[bne],
                        nodes[tne],
                    ]
                )


def prod(values: Iterable[int]) -> int:
    return reduce(lambda x, y: x * y, values, 1)


def first_and_has_more(values: Iterable[T]) -> Tuple[T, bool]:
    it = iter(values)
    first = next(it)
    try:
        next(it)
        return first, True
    except StopIteration:
        return first, False


def only(values: Iterable[T]) -> T:
    return next(iter(values))


def structured_cells(
    cellshape: Tuple[int, ...], pardim: int, nodemap: Optional[np.ndarray] = None
) -> FieldData[integer]:
    nodeshape = tuple(s + 1 for s in cellshape)
    ranges = [range(k) for k in cellshape]
    nidxs = [np.array(q) for q in zip(*product(*ranges))]
    eidxs = np.zeros((len(nidxs[0]), 2 ** len(nidxs)), dtype=int)
    if pardim == 1:
        eidxs[:, 0] = nidxs[0]
        eidxs[:, 1] = nidxs[0] + 1
    elif pardim == 2:
        i, j = nidxs
        eidxs[:, 0] = np.ravel_multi_index((i, j), nodeshape)
        eidxs[:, 1] = np.ravel_multi_index((i, j + 1), nodeshape)
        eidxs[:, 2] = np.ravel_multi_index((i + 1, j), nodeshape)
        eidxs[:, 3] = np.ravel_multi_index((i + 1, j + 1), nodeshape)
    elif pardim == 3:
        i, j, k = nidxs
        eidxs[:, 0] = np.ravel_multi_index((i, j, k), nodeshape)
        eidxs[:, 1] = np.ravel_multi_index((i, j, k + 1), nodeshape)
        eidxs[:, 2] = np.ravel_multi_index((i, j + 1, k), nodeshape)
        eidxs[:, 3] = np.ravel_multi_index((i, j + 1, k + 1), nodeshape)
        eidxs[:, 4] = np.ravel_multi_index((i + 1, j, k), nodeshape)
        eidxs[:, 5] = np.ravel_multi_index((i + 1, j, k + 1), nodeshape)
        eidxs[:, 6] = np.ravel_multi_index((i + 1, j + 1, k), nodeshape)
        eidxs[:, 7] = np.ravel_multi_index((i + 1, j + 1, k + 1), nodeshape)

    if nodemap is not None:
        eidxs = nodemap.flat[eidxs]

    return FieldData(eidxs)


def nodemap(
    shape: Tuple[int, ...], strides: Tuple[int, ...], periodic: Tuple[int, ...] = (), init: int = 0
) -> np.ndarray:
    indices = np.meshgrid(*(np.arange(s, dtype=int) for s in shape), indexing="ij")
    nodes = sum(i * s for i, s in zip(indices, strides)) + init
    assert isinstance(nodes, np.ndarray)
    for axis in periodic:
        nodes[_single_index(nodes.ndim, axis, -1)] = nodes[_single_index(nodes.ndim, axis, 0)]
    return nodes


def filename_generator(basename: Path, instantaneous: bool) -> Iterator[Path]:
    if instantaneous:
        yield basename
        return
    stem = basename.stem
    suffix = basename.suffix
    for i in count(1):
        yield basename.with_name(f"{stem}-{i}{suffix}")


def angular_mean(data: np.ndarray) -> np.ndarray:
    data = np.deg2rad(data)
    data = np.arctan2(np.mean(np.sin(data)), np.mean(np.cos(data)))
    return np.rad2deg(data)
