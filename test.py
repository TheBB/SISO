from multipledispatch import dispatch

from typing import Generic, TypeVar



T = TypeVar('T')
class TestClass(Generic[T]):
    pass


@dispatch(TestClass[int])
def myfunc(testvar):
    print('int')

@dispatch(TestClass[float])
def myfunc(testvar):
    print('float')


test = TestClass[int]()
myfunc(testvar)
