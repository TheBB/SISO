from __future__ import annotations

from typing_extensions import Protocol, TypeGuard, reveal_type


class A(Protocol):
    def is_b(self) -> TypeGuard[B]:
        ...
    def is_c(self) -> TypeGuard[C]:
        ...


class B(A):
    def is_b(self) -> TypeGuard[B]:
        return True
    def is_c(self) -> TypeGuard[C]:
        return False


class C(A):
    def is_b(self) -> TypeGuard[B]:
        return False
    def is_c(self) -> TypeGuard[C]:
        return True


def b_or_c(x: int) -> A:
    if x < 0:
        return B()
    return C()


b = b_or_c(-1)
if A.is_b(b):
    reveal_type(b)
